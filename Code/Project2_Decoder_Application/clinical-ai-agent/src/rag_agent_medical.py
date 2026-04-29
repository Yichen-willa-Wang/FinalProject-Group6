"""
Medical Follow-up RAG Agent
Specialized for patient follow-up with empathetic communication
"""

from rag_agent import HuggingFaceLLM, MedicalRAGAgent
from typing import Dict

class MedicalFollowupAgent(MedicalRAGAgent):
    """
    Enhanced RAG Agent for medical follow-up
    Integrates role-based prompting with RAG
    """
    
    def __init__(self, vectordb, llm, top_k=5, similarity_threshold=0.5):
        super().__init__(vectordb, llm, top_k, similarity_threshold)
        
        # Role definition
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
        Answer with role-based prompting
        
        Args:
            question: Patient question
            patient_context: Patient history (optional)
        """
        # Retrieve relevant documents
        relevant_docs = self.retriever.get_relevant_documents(question)
        
        if not relevant_docs:
            return {
                'answer': self._generate_no_context_response(question),
                'sources': [],
                'confidence': 0.0
            }
        
        # Build context
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            content_type = doc.metadata.get('content_type', 'text')
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', 'unknown')
            
            # Simplified source reference
            context_parts.append(f"[参考资料 {i}]\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # Build enhanced prompt
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
如果患者提到以下症状，立即回复：
"这可能是需要医生立即评估的情况。建议您尽快联系您的医生或前往医院就诊。"
- 视力突然明显下降
- 剧烈疼痛
- 严重出血  
- 发热或感染迹象

现在请回答患者的问题（直接给出答案，不要重复问题）："""
        
        return prompt
    
    def _post_process_answer(self, answer: str) -> str:
        """Post-process the answer"""
        
        # Remove common unwanted phrases
        unwanted_phrases = [
            "根据你提供的内容",
            "根据引用的内容",
            "根据上述材料",
            "根据文档"
        ]
        
        for phrase in unwanted_phrases:
            answer = answer.replace(phrase, "根据官方指南和材料")
        
        # Ensure concise
        if len(answer) > 350:
            # Truncate gracefully
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
