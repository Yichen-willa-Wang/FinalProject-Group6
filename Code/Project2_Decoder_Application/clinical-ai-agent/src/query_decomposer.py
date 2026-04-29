"""
Query Decomposition for Complex Questions
"""

class QueryDecomposer:
    """Decompose complex questions into sub-queries"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def decompose(self, question: str) -> list:
        """
        Decompose complex question into simpler sub-questions
        
        Example:
        Input: "近视手术后多久可以运动，需要注意什么？"
        Output: [
            "近视手术后多久可以运动？",
            "近视手术后运动需要注意什么？"
        ]
        """
        
        # Simple rule-based decomposition
        if '，' in question or '和' in question or '以及' in question:
            prompt = f"""将以下复杂问题分解为2-3个简单的子问题。
每个子问题单独一行，不要编号。

问题：{question}

子问题："""
            
            response = self.llm.generate(prompt, max_tokens=150)
            sub_questions = [q.strip() for q in response.split('\n') if q.strip()]
            return sub_questions[:3]  # Max 3 sub-questions
        
        return [question]  # Single question
