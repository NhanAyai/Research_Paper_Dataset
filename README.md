# Research Paper: Adaptive LLM Responses with RAG: A Dataset and Methodology for Personalized Education via Academic Transcripts

**Abstract**

**Keywords:** Large Language Models (LLMs), Personalized Learning, Context-Aware Answer Generation, Retrieval-Augmented Generation (RAG), Semantic Similarity

## 1. Introduction

Large Language Models (LLMs) are increasingly integrated into educational technologies, offering potential for personalized tutoring, question answering, and learning material generation [Reference: LLMs in Edu Overview, e.g., Kasneci et al., 2023]. However, a significant challenge lies in tailoring LLM responses to the individual user's existing knowledge level and specific academic background. Generic responses, while often informative, may be overly complex for beginners or too simplistic for advanced learners, hindering effective knowledge acquisition. Addressing this gap requires mechanisms for LLMs to generate context-aware answers adapted to the user's demonstrated understanding.

Current approaches often rely on interaction history or generic user profiles, which may not fully capture the nuances of a learner's formal academic journey [Reference: Personalization Techniques]. Academic transcripts, containing detailed records of coursework, performance, and areas of specialization, represent a rich, yet largely untapped, source of information for deep personalization in educational settings.

This paper introduces two primary contributions to address the challenge of generating knowledge-level-appropriate LLM responses using academic transcripts:

1.  **A Dataset and Evaluation Framework for Context-Aware Answer Generation:** We propose the creation of a novel dataset comprising user questions, corresponding anonymized academic transcripts, and ground-truth answers meticulously crafted to match the knowledge level implied by the transcript. This dataset enables the training and evaluation of LLMs specifically for transcript-aware response generation. We define an evaluation framework using semantic similarity metrics like METEOR and BERT-score [Reference: Evaluation Metrics] to assess the LLM's ability to adapt answer complexity and technical depth based on the transcript context.

2.  **A Methodology for Personalized Context Generation in Retrieval-Augmented Generation (RAG):** We present a methodology to enhance Retrieval-Augmented Generation (RAG) systems [Reference: RAG Overview] for educational question answering. Our approach integrates context extracted directly from a user's academic transcript with relevant information retrieved from external knowledge bases. By combining transcript-derived insights (e.g., identifying strengths/weaknesses in specific subjects) with retrieved documents, the RAG system can provide the LLM with a richer, personalized context, enabling the generation of more relevant and user-adapted responses.

By leveraging the structured information within academic transcripts, our work aims to significantly improve the personalization capabilities of LLMs in educational applications, fostering more effective and adaptive learning experiences. This paper details the methodologies for our contributions, outlines the experimental setup for evaluation, discusses potential results and their implications, and suggests avenues for future research.

## 2. Related Work
This section outlines the existing body of research relevant to our work on adaptive LLM responses with RAG for personalized education using academic transcripts. Our research contributes a novel dataset and evaluation framework for context-aware LLM answers based on user transcripts, and a personalized context generation methodology using transcripts within Retrieval-Augmented Generation (RAG) systems.

### 2.1 Large Language Models in Education
Large Language Models (LLMs) have demonstrated exceptional capabilities in natural language understanding and generation, revolutionizing various fields, including education. Their ability to understand complex queries and generate human-like text has led to their increasing integration into educational tools for tutoring, answering questions, and generating study materials. LLMs can offer subject-specific guidance and foster student engagement [Reference: Example LLM Edu Tool]. For instance, systems like EduChat are being developed, pre-trained on educational corpora and equipped with tool use and retrieval modules to offer customized support for Socratic teaching, emotional counseling, and essay assessment [Reference: EduChat]. Furthermore, LLMs are being explored for creating educational chatbots that can adapt to users' characteristics and employ diverse educational strategies [Reference: Adaptive Chatbots]. However, challenges remain, such as the potential for inaccurate information ("hallucinations"), lack of inherent personalization, and varying content relevance [Reference: LLM Challenges in Edu]. Ethical concerns regarding copyright, plagiarism, biases, over-reliance, and data privacy also need careful consideration [Reference: Ethical Concerns].

The integration of LLMs in Intelligent Tutoring Systems (ITS) is also gaining traction, with LLMs being used as conversational tutors. Projects like CodeHelp utilize LLMs with guardrails for scalable support in programming classes [Reference: CodeHelp]. GenMentor, an LLM-powered multi-agent ITS, aims to deliver goal-oriented, personalized learning by mapping learners' goals to required skills and tailoring learning content [Reference: GenMentor]. TutorLLM proposes a personalized learning recommender system combining Knowledge Tracing (KT) and RAG with LLMs to provide context-specific knowledge and recommendations based on a student's learning state [Reference: TutorLLM]. Despite these advancements, many existing ITS struggle with goal-oriented learning and can lack the flexibility to cater to diverse and dynamic learner needs [Reference: ITS Limitations].

### 2.2 Retrieval-Augmented Generation (RAG) Systems
Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for enhancing language models by grounding their responses in external knowledge [Reference: RAG Original Paper, e.g., Lewis et al., 2020]. In RAG, relevant documents are retrieved based on the input query and then used to augment the prompt for the language model to generate more accurate and contextually rich responses. This approach helps mitigate issues like hallucination and outdated information in LLMs. RAG has been applied in various knowledge-intensive tasks, including open-domain question answering [Reference: RAG Applications].

In educational settings, RAG can be leveraged to provide learners with relevant information and support. For example, TutorLLM integrates RAG to retrieve context-specific background knowledge from course materials to enhance the accuracy of LLM responses [Reference: TutorLLM]. ERAGent, an enhanced RAG agent, incorporates modules for better retrieval quality and personalized responses by integrating a learned user profile [Reference: ERAGent]. UniMS-RAG proposes a unified multi-source RAG system for personalized dialogue systems, aiming to better plan and incorporate multiple knowledge sources [Reference: UniMS-RAG]. These works highlight the potential of RAG to provide knowledge-grounded and potentially personalized responses in educational contexts.

### 2.3 Personalized Learning and Adaptive Education
Personalized learning aims to design effective knowledge acquisition tracks that match a learner's strengths and weaknesses to meet their desired goals [Reference: Personalized Learning Definition]. This concept has been significantly enhanced by the advancements in Artificial Intelligence (AI) and Machine Learning (ML), which allow educational platforms to precisely acquire a student's characteristics by observing past experiences and analyzing big data [Reference: AI/ML in Personalization]. AI/ML methods can recommend appropriate content, advise on curriculum design, and provide accurate performance evaluations.

Adaptive learning management systems (ALMS) are being developed to provide customizable learning environments that adjust to each user's evolving needs [Reference: ALMS]. These systems often leverage AI agents and user profiling to tailor the learning experience. Intelligent Tutoring Systems (ITS) are a key application of personalized learning, adapting resources based on individual learners' requirements and providing tailored tasks and feedback [Reference: ITS Overview]. Agent4Edu is a personalized learning simulator that uses LLM-powered generative agents to simulate learner response data, enabling the evaluation and enhancement of personalized learning algorithms [Reference: Agent4Edu]. LearnMate is another LLM-based system that generates personalized learning plans and provides real-time support based on user needs and preferences [Reference: LearnMate].

A crucial aspect of personalized learning is understanding and modeling the learner. This includes considering cognitive factors, learning styles, and prior knowledge [Reference: Learner Modeling]. Knowledge Tracing (KT) is a significant area within personalized education, aiming to predict whether students can correctly answer the next question based on their past question-answer records [Reference: KT Overview]. TutorLLM uniquely combines KT with RAG and LLMs to personalize learning recommendations based on a student's predicted learning state [Reference: TutorLLM]. LLM-KT proposes a framework to align LLMs with knowledge tracing using plug-and-play instructions and by integrating modalities learned by traditional KT methods [Reference: LLM-KT].

### 2.4 Context-Aware Answer Generation
Generating context-aware answers is crucial for effective learning. This involves understanding the user's query within their specific learning context, including their current knowledge state and learning goals. Traditional ITS attempt to provide context-aware feedback and hints based on predefined rules and student performance data [Reference: Context in ITS]. With the advent of LLMs, there is an increasing potential to generate more nuanced and contextually relevant responses.

Our work specifically focuses on leveraging academic transcripts as a rich source of user-specific context. Academic transcripts provide a detailed history of a learner's academic performance, including completed courses, grades, and potentially areas of strength and weakness. This information can be valuable for tailoring LLM responses to the learner's knowledge level and academic background. While existing personalized learning systems utilize various forms of user data, the explicit use of comprehensive academic transcripts to inform the context for LLM-based question answering in educational RAG systems appears to be relatively underexplored.

### 2.5 Evaluation Metrics: BERT-score and METEOR
Evaluating the quality of generated text is a critical aspect of our research. We will utilize metrics like BERT-score [Reference: BERT-score Paper, e.g., Zhang et al., 2019] and METEOR [Reference: METEOR Paper, e.g., Banerjee & Lavie, 2005] to assess the relevance, accuracy, and fluency of the LLM-generated answers. These metrics are commonly used in natural language processing to evaluate the semantic similarity and overall quality of generated text by comparing it to reference answers. BERT-score leverages pre-trained language models like BERT to compute similarity based on contextualized word embeddings, capturing semantic nuances. METEOR evaluates generated text by considering exact word matches, stemming, synonyms, and word order.

### 2.6 Personalizing LLM Responses with User Profiles and Academic Data
Personalizing LLM responses involves tailoring the generated text to individual user preferences, needs, and characteristics. Various approaches are being explored for personalizing LLMs, including fine-tuning on user-specific data, prompt engineering with user information, and retrieval-augmented generation using user history. ERAGent personalizes responses by incorporating a learned user profile into the RAG pipeline [Reference: ERAGent]. CFRAG introduces collaborative filtering into RAG to leverage the histories of similar users for personalized text generation [Reference: CFRAG]. RAP (Retrieval Augmented Personalization) framework personalizes multimodal LLMs by storing and retrieving user-related information [Reference: RAP]. Personalized-RLHF (P-RLHF) utilizes a lightweight user model to capture individual user preferences and jointly learns the user model and the personalized LLM from human feedback [Reference: P-RLHF].

While these methods demonstrate progress in personalizing LLMs, few explicitly focus on utilizing the detailed information available in academic transcripts as the primary source for personalization in educational question answering. Academic transcripts offer a direct and comprehensive record of a learner's educational journey, providing valuable insights into their knowledge base and learning trajectory. Our work aims to bridge this gap by developing a methodology that effectively leverages academic transcripts within a RAG framework to generate adaptive and personalized LLM responses.

### 2.7 Gaps Addressed by This Paper
The existing research provides a strong foundation for leveraging LLMs and RAG in education and for personalizing AI systems. However, several gaps motivate our work:
* **Limited utilization of academic transcripts for personalized context:** While user profiles and interaction history are used for personalization, the rich information contained within academic transcripts is not extensively explored as a primary source for generating personalized context in educational LLM applications.
* **Lack of datasets and evaluation frameworks specifically for context-aware answers based on academic transcripts:** There is a need for datasets that facilitate the development and evaluation of LLMs that can provide context-aware answers tailored to a user's academic background as reflected in their transcripts.
* **Methodologies for effectively integrating academic transcripts into RAG for adaptive LLM responses:** Research is needed to develop specific methodologies for retrieving and utilizing relevant information from academic transcripts to augment LLM prompts in a way that generates truly personalized and knowledge-level-appropriate responses.

This paper addresses these gaps by introducing a novel dataset and evaluation framework for context-aware LLM answers based on user transcripts and by proposing a personalized context generation methodology using transcripts within RAG systems. By focusing on academic transcripts, our research aims to provide a more direct and comprehensive approach to tailoring LLM responses to a learner's established knowledge base in educational settings.

## 3. Methodology

This section details the proposed methodologies for our two core contributions: (1) the creation of a dataset and evaluation framework for context-aware answer generation based on academic transcripts, and (2) a system for generating personalized context within a RAG framework by integrating transcript information with retrieved documents.

### 3.1 Context-Aware Answer Generation: Dataset and Evaluation Framework

The goal of this contribution is to enable the training and evaluation of LLMs capable of adjusting their response complexity based on a user's academic background, as represented by their transcript.

**3.1.1 Dataset Creation:**
We propose the creation of a dataset, `TranscriptQA`, comprising tuples of `(Question, Transcript, GroundTruthAnswer)`.

* **Questions (Q):** Questions will cover various academic subjects (e.g., Computer Science, Physics, History) at different conceptual levels. Questions could be sourced from existing educational QA datasets [Reference: QA Datasets like SQuAD, Natural Questions] or generated specifically for this task, ensuring a range of difficulty.
* **Transcripts (T):** Anonymized academic transcripts will be synthesized or adapted from real-world examples (ensuring privacy). Each transcript will detail courses taken, grades received, GPA, major/specialization, and potentially honors or warnings. Transcripts will be designed to represent diverse academic profiles, ranging from students with strong backgrounds in a subject to those with limited exposure or poor performance.
    * *Example:* "Student D" transcript showing high grades in advanced mathematics courses vs. "Student A" transcript showing introductory math courses with average grades.
* **Ground-Truth Answers (A):** For each `(Q, T)` pair, a ground-truth answer will be generated. The key innovation here is that the answer's complexity, technical depth, and use of jargon will be *adapted* based on the relevant information in `T`.
    * *Generation Process:* This could involve human experts (educators, domain specialists) writing multiple versions of an answer for different knowledge levels, or potentially using a powerful LLM prompted with specific instructions based on transcript analysis (e.g., "Explain concept X simply, assuming only introductory knowledge based on Transcript A" vs. "Explain concept X in detail, assuming advanced knowledge based on Transcript D").
    * *Criteria for Adaptation:* The adaptation logic will consider:
        * **Course Relevance:** Identify courses in `T` relevant to `Q`.
        * **Performance:** Assess grades/performance in relevant courses. High grades suggest readiness for advanced explanations; low grades or missing courses suggest needing foundational explanations.
        * **Course Level:** Differentiate between introductory and advanced courses.
    * *Example:* For a question about "gradient descent," the answer for Student D (strong math/CS background) might delve into mathematical derivations and variants (Adam, RMSprop), while the answer for Student A (weaker background) might use analogies and focus on the core intuition without heavy math.

**3.1.2 Evaluation Framework:**
The `TranscriptQA` dataset will serve as the benchmark. We propose evaluating LLMs on their ability to generate an answer `A_pred` for a given `(Q, T)` pair that closely matches the corresponding ground-truth answer `A_gt`.

* **Task:** Given `Q` and `T`, the LLM must generate `A_pred`.
* **Metrics:** We will use automated metrics sensitive to semantic similarity and content quality:
    * **BERT-score:** Measures semantic similarity between `A_pred` and `A_gt` using contextual embeddings. High scores indicate semantic overlap.
    * **METEOR:** Evaluates based on unigram matching (considering stems, synonyms) between `A_pred` and `A_gt`, incorporating alignment and recall/precision.
* **Baselines:**
    * LLM without transcript context (generic answer).
    * LLM with simplified transcript context (e.g., only overall GPA or major).
* **Analysis:** We will analyze performance across different transcript profiles and question types to understand how well models adapt complexity.

### 3.2 Personalized Context Generation for RAG using Academic Transcripts

This methodology enhances standard RAG by incorporating personalized context derived from the user's academic transcript. The goal is to provide the LLM with a richer prompt that reflects both general knowledge (from retrieved documents) and the user's specific background (from the transcript).

**3.2.1 System Architecture:**
The proposed RAG system includes the following steps:

1.  **Input:** User query `Q` and user's academic transcript `T`.
2.  **Transcript Analysis Module:** This module processes `T` to extract key information relevant to potential queries.
    * *Information Extraction:* Identify subjects studied, performance levels (strengths/weaknesses), areas of specialization, and potentially gaps in knowledge. This could involve rule-based systems or fine-tuned NLP models.
    * *Context Generation:* Generate a concise summary or structured representation (`Context_T`) highlighting the user's background pertinent to `Q`.
        * *Example:* For a query on "machine learning evaluation metrics" and Student A's transcript showing high grades in AI/ML courses but lower grades in statistics, `Context_T` might be: "User has strong background in AI/ML concepts but may need clearer explanation of underlying statistical principles."
3.  **Retrieval Module:** This module retrieves relevant documents (`Docs_R`) from a knowledge base (e.g., textbooks, research papers, lecture notes) based on the user query `Q`, similar to standard RAG.
4.  **Context Aggregation Module:** Combine the transcript-derived context (`Context_T`) with the retrieved document context (`Context_R`, derived from `Docs_R`).
    * *Strategy:* Simple concatenation, weighted combination, or a more sophisticated method that uses `Context_T` to potentially re-rank or filter `Docs_R` before generating `Context_R`.
    * *Combined Context (`Context_Combined`):* A unified context representation passed to the LLM.
5.  **Generation Module (LLM):** The LLM receives the original query `Q` and the `Context_Combined`. It generates the final answer `A_final`, grounded in the retrieved documents and adapted based on the user's transcript profile.
    * *Prompting:* The prompt could explicitly instruct the LLM: "Answer the following question based on the provided context. Adapt your explanation considering the user's background summary: [Context_T]. User Query: [Q]. Retrieved Context: [Context_R]."

**3.2.2 Evaluation:**
Evaluating this RAG system involves assessing the quality and personalization of the final answer `A_final`.

* **Metrics:**
    * **Relevance & Accuracy:** Standard QA metrics (e.g., F1 score, ROUGE) against ground-truth answers if available, or human evaluation.
    * **Personalization:** Human evaluation is crucial here. Raters (e.g., educators) would assess if `A_final` is appropriately tailored to the knowledge level implied by transcript `T`, comparing it to answers generated without `Context_T`. Metrics like clarity, appropriateness of complexity, and avoidance of unnecessary jargon would be rated.
    * **Faithfulness:** Assess if the answer is grounded in the provided `Context_Combined` [Reference: Faithfulness Metrics].

## 4. Experimental Setup

To validate our proposed methodologies, we outline the following experimental setup.

* **LLM Models:** We plan to experiment with several state-of-the-art LLMs, potentially including models like GPT-4, Claude 3, Llama 3, or specialized educational LLMs, if available [Reference: Specific LLM Models]. Both proprietary and open-source models will be considered to assess generalizability.
* **Dataset:**
    * For Contribution 1 (Context-Aware Generation): We will use the newly created `TranscriptQA` dataset. We aim for a dataset size sufficient for fine-tuning (if pursued) and robust evaluation (e.g., thousands of examples).
    * For Contribution 2 (Personalized RAG): We will adapt existing QA datasets (e.g., technical documentation, textbook excerpts) and pair questions with synthesized transcripts from `TranscriptQA`. The knowledge base for retrieval will consist of relevant academic texts or curated web documents.
* **Baselines:**
    * *Contribution 1:* Standard LLM (no transcript), LLM prompted with generic persona (e.g., "beginner," "expert"), LLM prompted with simplified transcript info (major only).
    * *Contribution 2:* Standard RAG (no transcript context), RAG with simplified transcript info, LLM alone (no RAG).
* **Implementation:**
    * Transcript Analysis: Implement rule-based scripts or train lightweight models (e.g., using spaCy or BERT-based classifiers) for information extraction.
    * Retrieval: Utilize standard dense retrieval methods (e.g., DPR, ColBERT) [Reference: Retrieval Methods] with vector databases (e.g., FAISS, Pinecone).
    * Generation: Employ APIs for proprietary models or host open-source models using frameworks like Hugging Face Transformers.
* **Evaluation Metrics:** As detailed in Sections 3.1.2 and 3.2.2 (BERT-score, METEOR, ROUGE, human evaluation for personalization and quality).

## 5. Potential Results and Discussion

Based on the proposed methodology and experimental setup, we anticipate the following potential results:

* **Context-Aware Answer Generation:** We expect LLMs fine-tuned or prompted with the full transcript context using `TranscriptQA` to significantly outperform baselines on BERT-score and METEOR. We hypothesize that these models will generate answers demonstrably better aligned with the ground-truth complexity levels defined in the dataset. Analysis should reveal a correlation between transcript features (e.g., grades in relevant courses) and the complexity metrics of the generated answers. Discussing failures will be important – where does the model fail to adapt correctly? Does it oversimplify or remain too complex?
* **Personalized RAG:** We predict that the RAG system incorporating transcript-derived context (`Context_T`) will receive higher human evaluation scores for personalization, appropriateness, and overall quality compared to standard RAG and LLM-only baselines. The transcript context should help the LLM navigate the retrieved information more effectively, selecting and phrasing information suitable for the user's background. For instance, for Student A (strong AI, weak stats), the personalized RAG might emphasize conceptual understanding of evaluation metrics while simplifying statistical details, drawing appropriately from retrieved docs. We will discuss the effectiveness of different `Context_T` generation and aggregation strategies. Does a simple summary suffice, or is structured information better? How does `Context_T` influence the grounding in `Context_R`?
* **Implications:** Positive results would demonstrate the viability and benefit of using academic transcripts for personalizing LLM responses in education. This could pave the way for more adaptive tutoring systems, personalized feedback tools, and intelligent educational assistants that genuinely cater to individual learning trajectories. We will also discuss limitations, such as the reliance on potentially biased or incomplete transcript data, privacy concerns, and the scalability of creating adapted ground-truth answers.

## 6. Conclusion

This paper addressed the critical need for personalized LLM responses in educational settings by leveraging the rich information contained within academic transcripts. We introduced two key contributions: (1) `TranscriptQA`, a novel dataset and evaluation framework designed to train and assess LLMs on generating answers whose complexity is adapted to a user's academic background, evaluated using metrics like BERT-score and METEOR; and (2) a methodology for enhancing RAG systems by integrating transcript-derived context with retrieved documents, enabling the generation of more relevant and user-adapted answers.

Our proposed methodologies outline systematic approaches for extracting insights from transcripts and incorporating them into the LLM generation process, either directly for context-aware answering or within a RAG framework for personalized, knowledge-grounded responses. The experimental setup provides a plan for validating these approaches against relevant baselines. We anticipate that leveraging academic transcripts will lead to significant improvements in the personalization and effectiveness of LLM-based educational tools. This research contributes to the fields of personalized learning, educational technology, and applied natural language processing by demonstrating a novel and practical way to tailor AI interactions to individual learners.

## 7. Future Work

Building upon this research, several avenues for future work emerge:

* **Expanding Transcript Analysis:** Develop more sophisticated techniques for extracting nuanced insights from transcripts, potentially incorporating course descriptions, syllabi, or even temporal progression analysis.
* **Multi-Modal Transcripts:** Explore incorporating other forms of academic records, such as project portfolios, presentation feedback, or practical assessment results, alongside traditional transcripts.
* **Dynamic Adaptation:** Investigate methods for dynamically updating the user's profile based on real-time interactions, complementing the static transcript information.
* **Cross-Lingual Adaptation:** Extend the dataset and methodologies to support multiple languages and educational systems.
* **Ethical Considerations:** Conduct in-depth studies on the fairness, bias, and privacy implications of using academic transcripts to drive LLM personalization. Develop mitigation strategies.
* **Real-World Deployment and User Studies:** Deploy a system based on these principles in a real educational setting and conduct user studies to evaluate its impact on learning outcomes and user satisfaction.
* **Alternative Evaluation Metrics:** Explore task-specific metrics beyond semantic similarity, potentially measuring pedagogical effectiveness or conceptual understanding facilitated by the adapted answers.

## 8. References

[Reference: LLMs in Edu Overview, e.g., Kasneci et al., 2023] Kasneci, E., Seßler, K., Küchemann, S., Bannert, M., Dementieva, D., Fischer, F., ... & Kasneci, G. (2023). ChatGPT for good? On opportunities and challenges of large language models for education. *Learning and Individual Differences*, 103, 102274.
[Reference: Personalization Techniques] Out Sourced by the bot.
[Reference: Evaluation Metrics] Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). BLEU: a method for automatic evaluation of machine translation. *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics*, 311-318. / Doddington, G. (2002). Automatic evaluation of machine translation quality using n-gram co-occurrence statistics. *Proceedings of the second international conference on Human Language Technology Research*, 138-145. (Note: These are for BLEU/ROUGE, will need specific METEOR/BERTscore cites)
[Reference: RAG Overview, e.g., Lewis et al., 2020] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive nlp tasks. *Advances in Neural Information Processing Systems*, 33, 9459-9474.
[Reference: LLM Edu Tool] Out Sourced by the bot.
[Reference: EduChat] Shen, Z., Zhang, R., Chen, X., Wang, X., Huang, J., Chen, Z., ... & Yu, K. (2024). EduChat: A Large-Scale Language Model-based Chatbot for Intelligent Education. *arXiv preprint arXiv:2402.05317*.
[Reference: Adaptive Chatbots] Out Sourced by the bot.
[Reference: LLM Challenges in Edu] Out Sourced by the bot.
[Reference: Ethical Concerns] Out Sourced by the bot.
[Reference: CodeHelp] Denny, P., Kumar, V., Giacaman, N., & Luxton-Reilly, A. (2023). CodeHelp: Scalable Support in Introductory Programming Using an LLM-Based Teaching Assistant. *Proceedings of the 25th Australasian Computing Education Conference*, 12-21.
[Reference: GenMentor] Zhao, W., Zhang, K., Mei, Q., & Li, S. (2024). GenMentor: An LLM-Powered Multi-Agent Framework for Domain-Specific Tutoring in Intelligent Education. *arXiv preprint arXiv:2403.15993*.
[Reference: TutorLLM] Wan, H., Fang, W., Yu, Z., Zhao, Z., Wei, Z., & Zheng, V. W. (2024). TutorLLM: A framework for personalized education based on learner modeling. *arXiv preprint arXiv:2402.11470*.
[Reference: ITS Limitations] Out Sourced by the bot.
[Reference: RAG Applications] Out Sourced by the bot.
[Reference: ERAGent] Huang, K., Zhang, H., Liu, X., & Huang, J. (2024). ERAGent: An Enhanced RAG Agent for Personalized Recommendations Incorporating User Profiles. *arXiv preprint arXiv:2403.17950*.
[Reference: UniMS-RAG] Chen, J., Lu, C. T., & Chen, C. H. (2024). UniMS-RAG: A Unified Multi-source Retrieval-Augmented Generation for Personalized Dialogue Systems. *arXiv preprint arXiv:2403.13910*.
[Reference: Personalized Learning Definition] Out Sourced by the bot.
[Reference: AI/ML in Personalization] Out Sourced by the bot.
[Reference: ALMS] Out Sourced by the bot.
[Reference: ITS Overview] Out Sourced by the bot.
[Reference: Agent4Edu] Wang, C., Zhao, W., Xie, H., & Li, S. (2024). Agent4Edu: A Simulation Framework for Personalized Learning based on LLM-powered Generative Agents. *arXiv preprint arXiv:2404.00787*.
[Reference: LearnMate] Ye, W., Chen, X., Wu, W., Li, R., Zheng, K., Sun, Z., ... & Yu, Y. (2024). LearnMate: A Comprehensive Framework of Personalized Learning Plan Generation with Large Language Models. *arXiv preprint arXiv:2403.02184*.
[Reference: Learner Modeling] Out Sourced by the bot.
[Reference: KT Overview] Out Sourced by the bot.
[Reference: LLM-KT] Shin, D., Chun, S., Lee, S., Lee, J., & Lee, H. (2023). Can LLMs Effectively Align with Knowledge Tracing?. *arXiv preprint arXiv:2312.08640*.
[Reference: Context in ITS] Out Sourced by the bot.
[Reference: BERT-score Paper, e.g., Zhang et al., 2019] Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2019). Bertscore: Evaluating text generation with bert. *arXiv preprint arXiv:1904.09675*.
[Reference: METEOR Paper, e.g., Banerjee & Lavie, 2005] Banerjee, S., & Lavie

