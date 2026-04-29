"""
Medical Follow-up RAG Test with DeepSeek-R1:1.5b
Uses existing MedicalFollowupAgent with optimized prompts
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import torch
import time
from typing import Dict
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("="*80)
print("🏥 Medical Follow-up RAG Test (DeepSeek-R1:1.5b + GPU)")
print("="*80)

# GPU check
print(f"\n💻 Hardware:")
print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"   CUDA: {torch.cuda.is_available()}")

# ============================================================================
# Ollama LLM Wrapper (replaces HuggingFaceLLM)
# ============================================================================

class OllamaLLM:
    """
    Ollama local LLM wrapper
    Compatible with MedicalFollowupAgent interface
    """
    
    def __init__(self, model_name="deepseek-r1:1.5b", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def generate(self, prompt: str, max_tokens=512, temperature=0.7) -> str:
        """
        Generate response using Ollama API
        Compatible with original HuggingFaceLLM interface
        """
        import requests
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9
            }
        }
        
        response = requests.post(
            self.api_url,
            json=payload,
            timeout=300  # 5 min timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '')
        else:
            raise Exception(f"Ollama API Error: {response.status_code} - {response.text}")

# ============================================================================
# Medical Follow-up Agent (from your existing code)
# ============================================================================

class MedicalFollowupAgent:
    """
    Enhanced RAG Agent for medical follow-up
    Uses role-based prompting with empathetic communication
    """
    
    def __init__(self, vectordb, llm, top_k=5, similarity_threshold=0.5):
        self.vectordb = vectordb
        self.llm = llm
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        self.retriever = vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": top_k,
                "score_threshold": similarity_threshold
            }
        )
        
        # System role definition
        self.system_role = """你是一个专业、谨慎且富有同理心的医疗随访AI助手。

核心职责：
- 回答患者关于手术、恢复、用药和注意事项的问题
- 语言简洁明了，避免专业术语堆砌
- 必要时用生活化语言和举例辅助说明
- 严格控制回答字数在150字以内，最多300字

重要原则：
1. 不进行诊断或治疗决策
2. 不编造医疗事实
3. 信息来源于权威知识库
4. 超出范围时建议咨询医生
5. "屈光手术"可称为"近视手术"
6. 不要说"根据你提供的内容"或"根据引用的内容"
"""
    
    def query(self, question: str, patient_context: str = "") -> Dict:
        """
        Answer patient question using RAG
        
        Args:
            question: Patient question
            patient_context: Optional patient history
            
        Returns:
            Dict with answer, sources, and confidence
        """
        # Retrieve relevant documents
        relevant_docs = self.retriever.get_relevant_documents(question)
        
        if not relevant_docs:
            return {
                'answer': self._generate_no_context_response(question),
                'sources': [],
                'confidence': 0.0
            }
        
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            context_parts.append(f"[参考资料 {i}]\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # Build medical prompt
        prompt = self._build_medical_prompt(question, context, patient_context)
        
        # Generate answer
        answer = self.llm.generate(prompt, max_tokens=400, temperature=0.7)
        
        # Post-process answer
        answer = self._post_process_answer(answer)
        
        # Prepare sources
        sources = [
            {
                'source': doc.metadata.get('source', 'unknown'),
                'page': doc.metadata.get('page', 'unknown'),
                'content_type': doc.metadata.get('content_type', 'text'),
                'excerpt': doc.page_content[:150]
            }
            for doc in relevant_docs
        ]
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': len(relevant_docs) / self.top_k
        }
    
    def _build_medical_prompt(self, question: str, context: str, patient_context: str = "") -> str:
        """Build medical follow-up prompt"""
        
        prompt = f"""{self.system_role}

{'患者信息：' + patient_context if patient_context else ''}

权威医疗知识库：
{context}

患者问题：{question}

回答要求：
1. 基于上述权威医疗知识库回答
2. 用简单易懂的语言，避免过多专业术语
3. 如需专业术语，用通俗语言解释
4. 信息来源说明为"根据官方指南和材料"
5. 如果知识库无相关内容，坦诚告知并建议咨询医生
6. 严格控制字数在150字左右，最多300字
7. 不要输出图片或价格信息
8. 遇到紧急症状（视力突然下降、剧烈疼痛、严重出血、发热）必须提醒立即就医

紧急情况识别：
如果患者提到以下症状,立即回复：
"这可能是需要医生立即评估的情况。建议您尽快联系您的医生或前往医院就诊。"
- 视力突然明显下降
- 剧烈疼痛
- 严重出血  
- 发热或感染迹象

现在请回答患者的问题（直接给出答案，不要重复问题）："""
        
        return prompt
    
    def _post_process_answer(self, answer: str) -> str:
        """Post-process the generated answer"""
        
        # Remove unwanted phrases
        unwanted_phrases = [
            "根据你提供的内容",
            "根据引用的内容",
            "根据上述材料",
            "根据文档"
        ]
        
        for phrase in unwanted_phrases:
            answer = answer.replace(phrase, "根据官方指南和材料")
        
        # Ensure concise (max 350 chars)
        if len(answer) > 350:
            sentences = answer.split('。')
            truncated = []
            length = 0
            for s in sentences:
                if length + len(s) < 300:
                    truncated.append(s)
                    length += len(s)
                else:
                    break
            answer = '。'.join(truncated) + '。'
        
        return answer.strip()
    
    def _generate_no_context_response(self, question: str) -> str:
        """Generate response when no relevant context found"""
        
        return """很抱歉，我在现有的医疗知识库中没有找到与您问题直接相关的信息。

建议：
1. 您可以换个方式描述您的问题
2. 如果是紧急情况，请立即联系您的医生
3. 您也可以咨询医院的专业医护人员

如果您有其他关于术前术后护理的问题，我很乐意帮助您。"""

# ============================================================================
# Test Questions
# ============================================================================

test_questions = [
    # Chinese medical questions
    {
        'question': '白内障手术后需要注意什么?',
        'patient_context': ''
    },
    {
        'question': '高血压患者的饮食建议有哪些?',
        'patient_context': ''
    },
    {
        'question': '我做完近视手术第三天了,眼睛有点干涩,这正常吗?',
        'patient_context': '患者30岁,刚做完LASIK手术'
    },
    # English medical questions
    {
        'question': 'What is refractive surgery?',
        'patient_context': ''
    },
    {
        'question': 'What is the difference between PRK and LASIK surgery?',
        'patient_context': ''
    },
    # Emergency situation test
    {
        'question': '我术后眼睛突然看不清了,很疼,怎么办?',
        'patient_context': '患者术后第2天'
    }
]

# ============================================================================
# Main Test Execution
# ============================================================================

print(f"\n📝 Total test questions: {len(test_questions)}\n")

# Step 1: Load embeddings on GPU
print("🔄 Step 1: Loading embeddings on GPU...")
start = time.time()
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",
    model_kwargs={'device': 'cuda'}
)
print(f"✅ Embeddings loaded in {time.time()-start:.2f}s")

# Step 2: Load vector database
print("\n🔄 Step 2: Loading vector database...")
start = time.time()
vectordb = Chroma(
    persist_directory="/home/ubuntu/data/vector_db",
    embedding_function=embeddings
)
chunk_count = vectordb._collection.count()
print(f"✅ Vector DB loaded in {time.time()-start:.2f}s")
print(f"   Total chunks: {chunk_count}")

# Step 3: Initialize Ollama LLM
print("\n🔄 Step 3: Initializing DeepSeek-R1:1.5b...")
llm = OllamaLLM(model_name="deepseek-r1:1.5b")

# Test Ollama connection
try:
    test_response = llm.generate("Hello", max_tokens=20)
    print(f"✅ DeepSeek ready: {test_response[:50]}...")
except Exception as e:
    print(f"❌ Ollama error: {e}")
    print("💡 Make sure Ollama is running: ollama serve &")
    sys.exit(1)

# Step 4: Initialize Medical Follow-up Agent
print("\n🔄 Step 4: Initializing Medical Follow-up Agent...")
agent = MedicalFollowupAgent(
    vectordb=vectordb,
    llm=llm,
    top_k=3,
    similarity_threshold=0.5
)
print("✅ Agent ready\n")

# Step 5: Run tests
print("="*80)
print("🧪 Running Medical Follow-up Tests")
print("="*80 + "\n")

results = []
total_start = time.time()

for i, test_case in enumerate(test_questions, 1):
    question = test_case['question']
    patient_context = test_case.get('patient_context', '')
    
    print(f"[Test {i}/{len(test_questions)}]")
    print(f"Question: {question}")
    if patient_context:
        print(f"Context: {patient_context}")
    print("-"*80)
    
    start_time = time.time()
    
    try:
        result = agent.query(question, patient_context)
        elapsed = time.time() - start_time
        
        answer = result['answer']
        sources = result['sources']
        confidence = result['confidence']
        
        print(f"⏱️  Response time: {elapsed:.2f}s")
        print(f"🎯 Confidence: {confidence:.2f}")
        print(f"\n📝 Answer ({len(answer)} chars):")
        print(f"{answer}\n")
        
        if sources:
            print(f"📚 Sources ({len(sources)}):")
            for j, src in enumerate(sources, 1):
                print(f"   {j}. {src['source']} (Page {src['page']})")
        else:
            print("📚 No sources found")
        
        print()
        
        results.append({
            'question': question,
            'time': elapsed,
            'status': 'Success',
            'answer_length': len(answer),
            'confidence': confidence,
            'num_sources': len(sources)
        })
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error after {elapsed:.2f}s: {str(e)}\n")
        
        results.append({
            'question': question,
            'time': elapsed,
            'status': 'Error',
            'answer_length': 0,
            'confidence': 0.0,
            'num_sources': 0
        })

total_elapsed = time.time() - total_start

# Summary
print("="*80)
print("📊 PERFORMANCE SUMMARY")
print("="*80)

success_count = len([r for r in results if r['status'] == 'Success'])
avg_time = sum(r['time'] for r in results) / len(results) if results else 0
min_time = min(r['time'] for r in results) if results else 0
max_time = max(r['time'] for r in results) if results else 0
avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
avg_answer_len = sum(r['answer_length'] for r in results) / len(results) if results else 0

print(f"\n📈 Results:")
print(f"   Total questions:    {len(results)}")
print(f"   Successful:         {success_count}")
print(f"   Failed:             {len(results) - success_count}")
print(f"   Success rate:       {success_count/len(results)*100:.1f}%")

print(f"\n⏱️  Performance:")
print(f"   Average time:       {avg_time:.2f}s per question")
print(f"   Fastest response:   {min_time:.2f}s")
print(f"   Slowest response:   {max_time:.2f}s")
print(f"   Total test time:    {total_elapsed:.2f}s ({total_elapsed/60:.1f} min)")

print(f"\n📝 Quality:")
print(f"   Average confidence: {avg_confidence:.2f}")
print(f"   Average answer length: {avg_answer_len:.0f} chars")

print(f"\n🔧 Configuration:")
print(f"   Model:              DeepSeek-R1:1.5b (934MB)")
print(f"   GPU:                {torch.cuda.get_device_name(0)}")
print(f"   Vector DB chunks:   {chunk_count}")
print(f"   Top-K retrieval:    3")
print(f"   Similarity threshold: 0.5")
print(f"   Embedding device:   CUDA")

print("\n" + "="*80)
print("✅ MEDICAL FOLLOW-UP RAG TEST COMPLETE!")
print("="*80)
