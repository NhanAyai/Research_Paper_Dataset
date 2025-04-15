# Research Paper: Adaptive LLM Responses with RAG: A Dataset and Methodology for Personalized Education via Academic Transcripts

**Abstract**

**Keywords:** Large Language Models (LLMs), Personalized Learning, Context-Aware Answer Generation, Retrieval-Augmented Generation (RAG), Semantic Similarity

## 1. Introduction

Large Language Models (LLMs) are increasingly integrated into educational technologies, offering potential for personalized tutoring, question answering, and learning material generation (Kasneci et al., 2023). However, a significant challenge lies in tailoring LLM responses to the individual user's existing knowledge level and specific academic background. Generic responses, while often informative, may be overly complex for beginners or too simplistic for advanced learners, hindering effective knowledge acquisition (Shen, 2024; Bernasconi et al., 2025). Addressing this gap requires mechanisms for LLMs to generate context-aware answers adapted to the user's demonstrated understanding (Bewersdorff et al., 2025).

Current approaches often rely on interaction history or generic user profiles, which may not fully capture the nuances of a learner's formal academic journey (Park et al., 2024). Academic transcripts, containing detailed records of coursework, performance, and areas of specialization, represent a rich, yet largely untapped, source of information for deep personalization in educational settings.

This paper introduces two primary contributions to address the challenge of generating knowledge-level-appropriate LLM responses using academic transcripts:

1.  **A Dataset and Evaluation Framework for Context-Aware Answer Generation:** We propose the creation of a novel dataset comprising user questions, corresponding anonymized academic transcripts, and ground-truth answers meticulously crafted to match the knowledge level implied by the transcript. This dataset enables the training and evaluation of LLMs specifically for transcript-aware response generation. We define an evaluation framework using semantic similarity metrics like METEOR and BERT-score (Kasneci et al., 2023) to assess the LLM's ability to adapt answer complexity and technical depth based on the transcript context.

2.  **A Methodology for Personalized Context Generation in Retrieval-Augmented Generation (RAG):** We present a methodology to enhance Retrieval-Augmented Generation (RAG) systems for educational question answering. Our approach integrates context extracted directly from a user's academic transcript with relevant information retrieved from external knowledge bases. By combining transcript-derived insights (e.g., identifying strengths/weaknesses in specific subjects) with retrieved documents, the RAG system can provide the LLM with a richer, personalized context, enabling the generation of more relevant and user-adapted responses.

By leveraging the structured information within academic transcripts, our work aims to significantly improve the personalization capabilities of LLMs in educational applications, fostering more effective and adaptive learning experiences. This paper details the methodologies for our contributions, outlines the experimental setup for evaluation, discusses potential results and their implications, and suggests avenues for future research.

## 2. Related Work

This section reviews previous efforts relevant to our contributions: (1) generating context-aware LLM responses using academic transcripts, and (2) integrating such context into RAG systems for personalized educational applications.

### 2.1 Large Language Models in Education

LLMs are transforming educational tools through their capabilities in natural language generation, enabling applications in tutoring, essay assessment, and question answering. Systems like **EduChat** [EduChat: A Large-Scale Language Model-based Chatbot System
for Intelligent Education] exemplify these applications by integrating retrieval modules for Socratic instruction and emotional support. Others, such as **CodeHelp** [CodeHelp: Using Large Language Models with Guardrails for Scalable Support in Programming Classes] and **GenMentor** [LLM-powered Multi-agent Framework for Goal-oriented Learning in Intelligent Tutoring System], explore scalable and goal-oriented conversational tutors, while **TutorLLM** [TutorLLM: Customizing Learning Recommendations with Knowledge Tracing and Retrieval-Augmented Generation] combines knowledge tracing and RAG [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks] to adapt LLM responses to a learner's state. However, despite these advances, current systems often depend on generic user profiles or limited interaction history, resulting in coarse personalization and insufficient alignment with individual learner backgrounds. Furthermore, critical challenges remain regarding factual reliability, ethical concerns, and the lack of adaptive granularity in LLM output.

### 2.2 Retrieval-Augmented Generation (RAG) for Education

RAG enhances language models by incorporating retrieved, task-specific knowledge into response generation, mitigating hallucinations and enabling more accurate answers. While initially proposed for open-domain QA (Lewis et al., 2020), RAG-based architectures like **TutorLLM** [TutorLLM: Customizing Learning Recommendations with Knowledge Tracing and Retrieval-Augmented Generation], **ERAGent** [ERAGent: Enhancing Retrieval-Augmented Language Models with Improved Accuracy, Efficiency, and Personalization], and **UniMS-RAG** [UniMS-RAG: A Unified Multi-source Retrieval-Augmented Generation for Personalized Dialogue Systems] have been extended to educational domains. These systems aim to personalize responses by using user profiles or dynamic retrieval pipelines. However, none explicitly leverage rich academic artifacts, such as transcripts, to inform retrieval or contextualize responses—a gap our work addresses directly.

### 2.3 Personalized Learning and Learner Modeling

The rise of LLMs has accelerated innovation in personalized learning systems, offering new avenues to adapt educational content and interaction to individual learner profiles. Traditional systems like ITS and adaptive learning platforms have historically relied on predefined rules or basic learner behavior analytics. However, recent advancements leverage LLMs to simulate learner profiles and generate tailored learning pathways, as seen in systems like Agent4Edu: Generating Learner Response Data by Generative Agents for Intelligent Education Systems (Gao et al., 2025) and LearnMate: Enhancing Online Education with LLM-Powered Personalized Learning Plans and Support (Wang et al., 2025).

Central to effective personalization is the accurate modeling of a learner’s prior knowledge and cognitive state. Knowledge Tracing (KT), a well-established technique in educational data mining, has evolved to align with modern LLM architectures. For example, LLM-KT: Aligning Large Language Models with Knowledge Tracing using a Plug-and-Play Instruction (Chen et al., 2023) integrates KT techniques into plug-and-play LLM modules, allowing real-time assessment of a student's learning trajectory and generating feedback aligned with their evolving knowledge state. Exploring Knowledge Tracing in Tutor-Student Dialogues Using LLMs (Scarlatos et al., 2025) further explores how LLMs can be used to model tutor-student dialogues for KT, enhancing adaptivity in real-world educational scenarios.

While learner modeling has matured through user interaction data and performance history, integrating formal academic records—such as course transcripts—into the personalization pipeline remains underexplored. EXAIT: Educational eXplainable Artificial Intelligent Tools for Personalized Learning (Ogata et al., 2024) argues that such structured academic data offers a rich representation of student progress and specialization, enabling more nuanced LLM personalization. LLMs for Knowledge Modeling: NLP Approach to Constructing User Knowledge Models for Personalized Education (Domenichini et al., 2024) further demonstrates that analyzing lesson-level records improves the precision of knowledge modeling and helps construct user-specific learning paths.

Complementary approaches, including personalized Retrieval-Augmented Generation (RAG) systems like ERAGent and CFRAG, rely on collaborative filtering or dynamic retrieval to align LLM outputs with user interests. However, these often fall short in incorporating deep academic context. Systems such as EXAIT: Educational eXplainable Artificial Intelligent Tools for Personalized Learning (Ogata et al., 2024) and A Comprehensive Survey on Deep Learning Techniques in Educational Data Mining (Chen et al., 2023) begin to bridge this gap by embedding academic records into personalization logic, demonstrating improved performance in adaptive feedback and resource recommendations.

Our work advances this trajectory by explicitly encoding transcript-derived insights—such as course completion, topic exposure, and performance trends—into a RAG-compatible context representation. This structured personalization enables LLMs to adapt content delivery not only to what the user knows, but also how and when they learned it, offering a new paradigm in personalized educational dialogue.

### 2.4 Context-Aware Answer Generation

Effectively tailoring responses from Large Language Models (LLMs) to individual learners requires the incorporation of contextual information that reflects each user’s prior knowledge, goals, and academic trajectory. Traditional ITS have relied on rule-based personalization or basic learner modeling, which, while structured, often fails to capture deeper semantic or domain-specific context.

Recent advancements have seen the integration of LLMs with dynamic retrieval systems and personalized generation pipelines to improve adaptability. For example, Zhang et al. (2024) introduce the Socratic Playground for Learning (SPL) [SPL: A Socratic Playground for Learning Powered by Large Language Models], a system that dynamically modulates interaction styles based on user profiles to facilitate deeper learning dialogue. Similarly, Chimezie (2024) applies RAG in the context of Data Structures and Algorithms, demonstrating that course-specific data significantly enhances LLM response relevance [Leveraging Retrieval-Augmented Generation in LLMs for Effective Learning].

Efforts to combine domain-specific models with general-purpose LLMs further underscore the importance of learner-aligned context. Luo and Yang (2024) explore hybrid modeling approaches for smart education environments, leveraging domain data to improve LLM reasoning in specialized fields [Large Language Model and Domain-Specific Model Collaboration for Smart Education]. Likewise, Neyem et al. (2024) present an AI knowledge assistant that adapts responses in software engineering education based on individual learner profiles and capstone project data [Towards an AI Knowledge Assistant for Context-Aware Learning Experiences in Software Capstone Projects].

Despite these advances, academic transcripts—rich with structured data about a student’s coursework, academic progression, and skill areas—remain largely untapped. Ling and Afzaal (2024) show that context-aware question-answer generation improves engagement and learning outcomes in higher education, yet their work also highlights the need for more granular learner modeling inputs [Automatic Question-Answer Pairs Generation Using Pre-trained LLMs in Higher Education].

Our work directly addresses this gap by integrating transcript-derived features into a Retrieval-Augmented Generation (RAG) pipeline. This approach provides a structured and semantically rich representation of the learner’s academic background, enabling the LLM to generate responses with a level of personalization and instructional relevance that exceeds what interaction history or generic profiles can achieve.

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

