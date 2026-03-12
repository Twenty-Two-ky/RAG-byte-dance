import chromadb
import os
import json
import uuid
import sys
import logging
from datetime import datetime
import google.generativeai as genai
from PIL import Image

# LangChain组件（新增：替代自研TextSplitter，引入混合检索+重排序）
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.prompts import PromptManager

logger = logging.getLogger(__name__)

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'vector_db')
DOC_PATH = os.path.join(BASE_DIR, 'data', 'raw_files') 

if not os.path.exists(DOC_PATH):
    os.makedirs(DOC_PATH)

class RAGEngine:
    def __init__(self, api_key):
        if not api_key: raise ValueError("RAG Engine 需要 API Key")
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.embedding_fn = GeminiEmbeddingFunction(api_key)
        self.api_key = api_key

        self.history_coll = self.client.get_or_create_collection(name="history_cases", embedding_function=self.embedding_fn)
        self.knowledge_coll = self.client.get_or_create_collection(name="company_knowledge", embedding_function=self.embedding_fn)

        # === 新增: LangChain文本切片器（替代自研TextSplitter的80行代码）===
        # separators按优先级递减: 先在双换行切，再单换行，再句号等标点
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )

        # === 新增: BM25索引（用于混合检索的关键词检索路）===
        self._bm25_index = None
        self._bm25_corpus = []    # 原始文本列表
        self._bm25_doc_ids = []   # 对应的文档ID
        self._build_bm25_index()

        # === 新增: BGE-Reranker（二次精排模型）===
        # 首次运行会自动从HuggingFace下载（约400MB），后续使用本地缓存
        try:
            self.reranker = CrossEncoder('BAAI/bge-reranker-base')
            logger.info("BGE-Reranker 加载成功")
        except Exception as e:
            logger.warning(f"BGE-Reranker 加载失败，将跳过重排序: {e}")
            self.reranker = None

    # ==================== 新增: 混合检索相关私有方法 ====================

    def _build_bm25_index(self):
        """
        构建BM25倒排索引。

        从knowledge_coll中取出所有文档，对每个文档做分词（按单字切分），
        构建BM25Okapi索引。

        调用时机:
        1. __init__初始化时
        2. add_knowledge()添加新文档后（需rebuild）
        """
        try:
            all_data = self.knowledge_coll.get()
            if all_data['documents']:
                self._bm25_corpus = all_data['documents']
                self._bm25_doc_ids = all_data['ids']
                # 中文分词: 按单字切分（简单有效，无需jieba依赖）
                tokenized = [list(doc) for doc in self._bm25_corpus]
                self._bm25_index = BM25Okapi(tokenized)
                logger.info(f"BM25索引构建完成，共 {len(self._bm25_corpus)} 个文档片段")
            else:
                self._bm25_index = None
                logger.info("知识库为空，跳过BM25索引构建")
        except Exception as e:
            logger.warning(f"BM25索引构建失败: {e}")
            self._bm25_index = None

    def _bm25_search(self, query, top_k=10):
        """
        BM25关键词检索。

        Args:
            query: 查询文本
            top_k: 返回条数

        Returns:
            list[dict]: [{"id": chunk_id, "content": text, "score": float}, ...]
        """
        if not self._bm25_index:
            return []

        tokenized_query = list(query)  # 与建索引时相同的分词方式
        scores = self._bm25_index.get_scores(tokenized_query)

        # 取Top-K（按分数降序）
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 过滤零分结果
                results.append({
                    "id": self._bm25_doc_ids[idx],
                    "content": self._bm25_corpus[idx],
                    "score": float(scores[idx])
                })
        return results

    def _rerank(self, query, candidates, top_k=3):
        """
        使用BGE-Reranker对候选文档做精排。

        原理: Cross-Encoder把(query, document)拼接后联合编码，
        计算相关性分数，精度远高于向量检索的Bi-Encoder。
        如果Reranker未加载，退化为直接截取Top-K（容错降级）。

        Args:
            query: 查询文本
            candidates: list[dict]，每个元素需包含 "content" 字段
            top_k: 返回条数

        Returns:
            list[dict]: 重排序后的Top-K结果
        """
        if not self.reranker or not candidates:
            return candidates[:top_k]

        # 构造 (query, document) 对
        pairs = [[query, c["content"]] for c in candidates]

        # Cross-Encoder打分
        scores = self.reranker.predict(pairs)

        # 按分数降序排列
        sorted_indices = np.argsort(scores)[::-1][:top_k]

        reranked = []
        for idx in sorted_indices:
            candidate = candidates[idx].copy()
            candidate["rerank_score"] = float(scores[idx])
            reranked.append(candidate)

        return reranked

    # ==================== 原有方法 ====================

    def _save_raw_file(self, file_obj, filename):
        safe_name = f"{uuid.uuid4().hex[:8]}_{filename}"
        file_path = os.path.join(DOC_PATH, safe_name)
        file_obj.seek(0)
        with open(file_path, "wb") as f:
            f.write(file_obj.read())
        return file_path

    def parse_file_content(self, file_obj, file_type, model_name="models/gemini-1.5-flash"):
        """利用 AI 解析图片/PDF 内容为文本"""
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(model_name)
            
            content_part = []
            file_obj.seek(0)
            
            prompt = PromptManager.MULTIMODAL_PARSE_PROMPT
            
            if "image" in file_type:
                img = Image.open(file_obj)
                content_part = [prompt, img]
            elif "pdf" in file_type:
                file_bytes = file_obj.read()
                content_part = [prompt, {"mime_type": "application/pdf", "data": file_bytes}]
            else:
                return file_obj.read().decode('utf-8')
            
            resp = model.generate_content(content_part)
            return resp.text
        except Exception as e:
            return f"[解析失败] {str(e)}"

    def add_knowledge(self, file_obj, summary="", content_text=None, model_name="models/gemini-1.5-flash"):
        """支持 Chunking 切片存储"""
        saved_path = self._save_raw_file(file_obj, file_obj.name)
        
        final_content = ""
        if content_text:
            final_content = content_text
        else:
            if "text" in file_obj.type or "md" in file_obj.name:
                file_obj.seek(0)
                final_content = file_obj.getvalue().decode("utf-8")
            else:
                final_content = self.parse_file_content(file_obj, file_obj.type, model_name)

        # 执行切片（使用LangChain RecursiveCharacterTextSplitter，替代自研TextSplitter）
        chunks = self.text_splitter.split_text(final_content)
        
        file_doc_id = str(uuid.uuid4())
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        ids = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_doc_id}_chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "doc_id": file_doc_id,
                "source": file_obj.name,
                "summary": summary if summary else "暂无摘要",
                "file_path": saved_path,
                "chunk_index": i,
                "type": "spec",
                "date": current_time
            })

        self.knowledge_coll.add(documents=documents, metadatas=metadatas, ids=ids)

        # 新增: 添加文档后重建BM25索引，使新文档可被关键词检索到
        self._build_bm25_index()

    def add_history_case(self, prd_text, final_json, summary=""):
        if isinstance(final_json, (dict, list)):
            final_json = json.dumps(final_json, ensure_ascii=False)
        file_doc_id = str(uuid.uuid4())
        self.history_coll.add(
            documents=[prd_text],
            metadatas=[{"doc_id": file_doc_id, "answer": final_json, "source": summary, "summary": summary, "type": "history", "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "file_path": "N/A"}],
            ids=[file_doc_id]
        )

    def list_documents(self, collection_type="knowledge"):
        coll = self.history_coll if collection_type == "history" else self.knowledge_coll
        data = coll.get()
        unique_docs = {}
        if data['ids']:
            for i, _ in enumerate(data['ids']):
                meta = data['metadatas'][i]
                doc_id = meta.get('doc_id')
                if not doc_id: # 兼容旧数据
                    unique_docs[data['ids'][i]] = {"ID": data['ids'][i], "文件名/标题": meta.get('source', 'unknown'), "AI摘要": meta.get('summary', '-'), "类型": "历史用例" if collection_type == "history" else "技术文档", "录入时间": meta.get('date', '-'), "原始路径": meta.get('file_path', 'N/A')}
                    continue
                if doc_id not in unique_docs:
                    unique_docs[doc_id] = {"ID": doc_id, "文件名/标题": meta.get('source', 'unknown'), "AI摘要": meta.get('summary', '-'), "类型": "历史用例" if collection_type == "history" else "技术文档", "录入时间": meta.get('date', '-'), "原始路径": meta.get('file_path', 'N/A')}
        return list(unique_docs.values())

    def get_doc_content(self, file_path, doc_id=None, collection_type="knowledge"):
        coll = self.history_coll if collection_type == "history" else self.knowledge_coll
        if doc_id:
            try:
                item = coll.get(where={"doc_id": doc_id}, limit=1)
                if not item['ids']: item = coll.get(ids=[doc_id], limit=1)
                if item['documents'] and item['documents'][0]:
                    if collection_type == "history" and item['metadatas']:
                        json_str = item['metadatas'][0].get('answer', '{}')
                        try:
                            parsed = json.loads(json_str)
                            return json.dumps(parsed, indent=2, ensure_ascii=False)
                        except: return json_str
                    return item['documents'][0]
            except Exception: pass
        if os.path.exists(file_path):
            try:
                if file_path.endswith(('.txt', '.md', '.json', '.yaml', '.csv')):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: return f.read()
            except: pass
        return "无法获取文档内容"

    def delete_document(self, doc_id, collection_type="knowledge"):
        coll = self.history_coll if collection_type == "history" else self.knowledge_coll
        items = coll.get(where={"doc_id": doc_id}, limit=1)
        if not items['ids']: items = coll.get(ids=[doc_id], limit=1)
        if items['metadatas']:
            path = items['metadatas'][0].get('file_path')
            if path and path != "N/A" and os.path.exists(path):
                try: os.remove(path)
                except: pass
        coll.delete(where={"doc_id": doc_id})
        coll.delete(ids=[doc_id])

    def hybrid_search(self, query, top_k=3, vector_weight=0.7, bm25_weight=0.3):
        """
        混合检索: 向量检索 + BM25关键词检索 + Re-ranking精排。

        这是对外暴露的新接口，供Agent工具调用。
        原有的 search_context() 也在内部改用此方法。

        流程:
        1. 向量检索 Top-10 (语义相似)
        2. BM25检索 Top-10 (关键词匹配)
        3. RRF(Reciprocal Rank Fusion)融合 → Top-6
        4. BGE-Reranker精排 → Top-K

        Args:
            query: 查询文本
            top_k: 最终返回条数
            vector_weight: 向量检索权重 (默认0.7)
            bm25_weight: BM25检索权重 (默认0.3)

        Returns:
            list[dict]: [{"id", "content", "score", "metadata"}, ...]
        """
        recall_k = max(top_k * 3, 10)  # 召回阶段取更多候选

        # === 路径1: 向量检索 ===
        vector_results = []
        try:
            res = self.knowledge_coll.query(query_texts=[query], n_results=recall_k)
            if res['documents'] and res['documents'][0]:
                for i, doc in enumerate(res['documents'][0]):
                    distance = res['distances'][0][i] if res.get('distances') else 1.0
                    vector_results.append({
                        "id": res['ids'][0][i],
                        "content": doc,
                        "metadata": res['metadatas'][0][i],
                        "score": 1.0 / (1.0 + distance)  # 距离转相似度
                    })
        except Exception as e:
            logger.warning(f"向量检索失败: {e}")

        # === 路径2: BM25检索 ===
        bm25_results = self._bm25_search(query, top_k=recall_k)

        # === 融合: Reciprocal Rank Fusion (RRF) ===
        # RRF公式: score = weight / (k + rank)，k=60是论文推荐的经验值
        fused_scores = {}

        for rank, item in enumerate(vector_results):
            doc_id = item["id"]
            fused_scores[doc_id] = {
                "content": item["content"],
                "metadata": item.get("metadata", {}),
                "score": vector_weight / (60 + rank)
            }

        for rank, item in enumerate(bm25_results):
            doc_id = item["id"]
            if doc_id in fused_scores:
                # 两路都召回了同一文档 → 分数叠加（更相关）
                fused_scores[doc_id]["score"] += bm25_weight / (60 + rank)
            else:
                fused_scores[doc_id] = {
                    "content": item["content"],
                    "metadata": {},
                    "score": bm25_weight / (60 + rank)
                }

        # 按融合分数排序，取Top候选送去Rerank
        sorted_candidates = sorted(
            [{"id": k, **v} for k, v in fused_scores.items()],
            key=lambda x: x["score"],
            reverse=True
        )[:top_k * 2]  # 给Reranker多一些候选

        # === 精排: BGE-Reranker ===
        final_results = self._rerank(query, sorted_candidates, top_k=top_k)

        return final_results

    def search_context(self, query, use_history=True, use_knowledge=True):
        """
        检索知识库上下文。

        接口签名和返回格式与旧版完全一致（向后兼容，main.py无需改动）。
        内部改用混合检索(hybrid_search)替代纯向量检索。
        如果混合检索失败，自动降级到纯向量检索。
        """
        context_parts = []
        sources = []

        # === 技术规范检索（改用混合检索: 向量+BM25+Rerank）===
        if use_knowledge:
            try:
                results = self.hybrid_search(query, top_k=3)
                for item in results:
                    src = item.get("metadata", {}).get("source", "unknown")
                    context_parts.append(f"【技术规范片段 ({src})】:\n...{item['content']}...")
                    sources.append(f"📚 {src}")
            except Exception as e:
                logger.warning(f"混合检索失败，降级到纯向量检索: {e}")
                # 降级兜底: 退回原来的纯向量检索，保证不影响用户使用
                res_k = self.knowledge_coll.query(query_texts=[query], n_results=3)
                if res_k['documents'] and res_k['documents'][0]:
                    for i, doc in enumerate(res_k['documents'][0]):
                        meta = res_k['metadatas'][0][i]
                        src = meta.get('source', 'unknown')
                        context_parts.append(f"【技术规范片段 ({src})】:\n...{doc}...")
                        sources.append(f"📚 {src}")

        # === 历史案例检索（保持纯向量，因为数据量小，BM25意义不大）===
        if use_history:
            res_h = self.history_coll.query(query_texts=[query], n_results=1)
            if res_h['documents'] and res_h['documents'][0]:
                for i, doc in enumerate(res_h['documents'][0]):
                    summary = res_h['metadatas'][0][i].get('summary', '历史案例')
                    ans = res_h['metadatas'][0][i].get('answer', '')
                    context_parts.append(f"【参考历史 ({summary})】:\n参考用例:{ans[:800]}...")
                    sources.append(f"🕰️ {summary}")

        return "\n\n<<<RAG_SEP>>>\n\n".join(context_parts), list(set(sources))

class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, api_key):
        genai.configure(api_key=api_key)

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=input,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            # 改进: 裸except → 明确Exception + 日志记录
            logger.error(f"Embedding生成失败: {e}")
            return [[0.0] * 768 for _ in input]