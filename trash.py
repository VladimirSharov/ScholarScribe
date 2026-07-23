'''
Original format:

Proccessing field (title+abstract, trancation; one hot encoded tags list):
    Sample:
    
'''
'''
control token for Roberta beginning, end?
extract weight from head of xlm roberta
print last layer coef and bias
coef low, bias high. check if it so, (bad?) last layer, load checkpoint
041025
Next explaining reason why, check inside the package.
Next iteration, improve script adding sample of it work before preprocessing to see if it working.
Language not working, then if so, do else case for input text.
'''
'''
In Microsoft Word, you can reveal special characters (including non-breaking spaces, tabs, paragraph marks, etc.) by enabling "Show/Hide ¶" (Ctrl+Shift+8 or click the ¶ button in the Home tab). This will display:

    Non-breaking space (U+00A0) as a small ° or ␠ (depending on Word version)

    Regular space as a dot ·

    Tabs as →

    Paragraph marks as ¶

Viewing Hidden Characters in VS Code

In VS Code, you can:

    Press Ctrl+F (Find) → Enable "Use Regular Expression" (Alt+R) → Search for \xA0 (for non-breaking space).

    Or install the "Render Whitespace" extension to visualize hidden chars.

    Alternatively, enable whitespace rendering in settings ("editor.renderWhitespace": "all").

Common "Invisible" Symbols in Word Documents

Word uses many hidden formatting symbols, including:
Symbol	Unicode	Description
·	U+0020	Regular space
° / ␠	U+00A0	Non-breaking space (prevents line breaks)
¶	U+00B6	Paragraph mark (Enter key)
→	U+2192	Tab character
¬	U+00AC	Optional hyphen (hidden unless needed)
¤	U+F0B7	Cell marker in tables
Do You Need to Know All These Symbols?

No! For most Word users, only a few matter:

    Non-breaking space (Ctrl+Shift+Space) → Keeps words together (e.g., "Mr. Smith").

    Non-breaking hyphen (Ctrl+Shift+-) → Prevents line breaks in phrases like "COVID‑19".

    Paragraph vs. Line Break (Shift+Enter) → ¶ vs. ↵.

How to Find/Replace Special Characters in Word

    Find (Ctrl+H) → Type ^s (non-breaking space) or ^t (tab).

    For Unicode chars, search ^u00A0 (non-breaking space).

Would you like a list of the most useful ones for editing?
'''
'''
Good question.

A **study line** typically refers to a **specialization or track within a broader degree program**—like a focused path of courses and research in a particular area. It’s commonly used in European universities, especially in Finland and other Nordic countries.

### Examples:
- In a **Computer Science** degree:
  - Study lines might include *Artificial Intelligence*, *Software Engineering*, *Human-Computer Interaction*, etc.
  
- In an **Engineering** degree:
  - Study lines might include *Signal Processing*, *Embedded Systems*, etc.

---

### How it differs from "domain":
- **Domain** is broader and more general. It refers to the **field of study or application area** (e.g., education, healthcare, finance, NLP).
- **Study line** is a **structured educational path** within a degree.

---

So, if your thesis is part of a study line, you’d refer to that in your thesis metadata or introduction, like:

> *This work was conducted as part of the [Machine Learning] study line within the [Computer Science] Master's program at [University Name].*

Would you like help phrasing how to describe your study line in your thesis introduction?
This thesis explores metadata-based trend analysis in multilingual academic texts, focusing on theses from the University of Jyväskylä written in Finnish and English. Using natural language processing techniques, the study investigates how metadata—such as abstracts, titles, and tags—can help detect research trends and assist in academic content organization.
Order in information is older than computing. Long before digital systems, libraries developed classification methods to store, retrieve, and organize knowledge. Systems like the Dewey Decimal Classification (DDC), Library of Congress Classification (LCC), and metadata standards like MARC (Machine-Readable Cataloging) defined how knowledge could be labeled and navigated. These early systems emphasized structure—books were tagged with subjects, authors, languages, and dates, not only for shelving but for discovery.

As information systems evolved, digital metadata frameworks emerged. One influential standard is the Dublin Core Metadata Initiative (DCMI), which introduced a simple, universal set of elements—title, creator, subject, description, and so on—adaptable across digital libraries. In academic repositories, such as those storing university theses, similar metadata schemas are used. These fields are minimal but structured: they reflect what is essential for identification and classification.

Metadata exists not for aesthetic purposes, but for usability. It enables search, indexing, recommendation, and large-scale analysis. Unlike full texts, which require intensive parsing, metadata offers concise, digestible context. In the digital age, it acts as both a pointer and a summary, bridging human organization and machine understanding.
Knowledge accumulates, multiplies, fragments. Without order, it overwhelms. The earliest libraries-imposed systems- taxonomies, classifications- not just to store books, but to impose structure on meaning. Today, digital repositories inherit this challenge. What was once the card catalog becoming metadata: compact, invisible, but powerful. In thesis databases, metadata like titles, abstracts, departments, and tags offers a sparse but structured mirror of the research landscape. Unlike full texts, which require expensive computation to parse, metadata is minimalistic and machine-readable. From this constraint emerges opportunity: patterns, correlations, even academic trends. These patterns don’t come from divine design or instant inspiration- they emerge from repetition, interaction, structure. In this chapter, we explore the systems and models that extract such structure from language, beginning with how machine learning reshaped the way we interpret text.
Suggested Subsections:

    3.1 Research Design and Goals (state classification task + why it matters)

    3.2 Data Collection (from thesis databases; cite sources)

    3.3 Data Preprocessing (cleaning, normalization, tokenization)

    3.4 Feature Engineering (you already wrote this well)

    3.5 Models Used (XLM-R and LLaMA, and why those were selected)

    3.6 Training Setup and Parameters (what libraries, how long, what metrics)

Optional but nice:

    3.7 Limitations of the Approach

    (base) [sharovv@calaf ScholarScribe]$ /home/sharovv/mambaforge/bin/python /home/sharovv/ScholarScribe/data_validation/tag_evaluation_v3.py
Analyzing tag similarity...
Loading sentence transformer model...
modules.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 229/229 [00:00<00:00, 983kB/s]
config_sentence_transformers.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 122/122 [00:00<00:00, 1.25MB/s]
README.md: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.89k/3.89k [00:00<00:00, 43.0MB/s]
sentence_bert_config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 53.0/53.0 [00:00<00:00, 642kB/s]
config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 645/645 [00:00<00:00, 7.51MB/s]
model.safetensors: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 471M/471M [00:04<00:00, 115MB/s]
tokenizer_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 480/480 [00:00<00:00, 4.96MB/s]
tokenizer.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9.08M/9.08M [00:00<00:00, 30.5MB/s]
special_tokens_map.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 239/239 [00:00<00:00, 2.78MB/s]
config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 190/190 [00:00<00:00, 2.11MB/s]
Processing 1766 items...
Processing item 1/1766...
Processing item 101/1766...
Processing item 201/1766...
Processing item 301/1766...
Processing item 401/1766...
Processing item 501/1766...
Processing item 601/1766...
Processing item 701/1766...
Processing item 801/1766...
Processing item 901/1766...
Processing item 1001/1766...
Processing item 1101/1766...
Processing item 1201/1766...
Processing item 1301/1766...
Processing item 1401/1766...
Processing item 1501/1766...
Processing item 1601/1766...
Processing item 1701/1766...
Visualizing results...
Results saved to evaluation_results/ directory
Analyzing tag tokens...
Analysis complete!

Sample results (5 items):
     language  emd_score  exact_match_f1  \
618   english   0.085525        0.375000   
997   english   0.138941        0.303030   
1565  english   0.139315        0.111111   
653   english   0.167943        0.090909   
924   unknown   0.187683        0.000000   

                                          original_tags  \
618   [tilintarkastus, standardit, pienyritykset, IS...   
997   [karjalan kieli, samaistuminen, diskurssi, tun...   
1565  [opetus, ikääntyneet, vanhukset, mobiililaitte...   
653   [luottolaitokset, tilinpäätös, konsernit, IFRS...   
924   [pariterapia, ryhmäterapia, tunteet, parisuhde...   

                                         generated_tags  
618   [tilintarkastus, ISA-standardit, pientyöllisyy...  
997   [karjalaiset, vähemmistöt, identiteetti, ident...  
1565  [ikäihmiset, teknologia, digitaalinen muutos, ...  
653   [luottolaitos, tilinpäätös, IFRS, konsernitili...  
924   [Emotionally Focused Therapy, Couple Therapy, ...  


Results saved to evaluation_results/ directory
Analyzing tag tokens...
Analysis complete!

Sample results (5 items):
     language  emd_score  exact_match_f1  \
832   finnish   0.182185        0.375000   
1700  unknown   0.237748        0.363636   
391   finnish   0.249802        0.230769   
992   english   0.307936        0.000000   
987   english   0.349567        0.000000   

                                          original_tags  \
832   [kelpoisuus, opettajankoulutus, opettajat, val...   
1700  [kunnat, tieto- ja viestintätekniikka, johtami...   
391   [elektroninen urheilu, vuorovaikutus, valmennu...   
992   [julkinen keskustelu, turvallisuus, parlamenti...   
987   [informaali oppiminen, oppiminen, kielen oppim...   

                                         generated_tags  
832   [maahanmuutto, koulutus, opettajat, valmistava...  
1700  [ICT-projektit, kunta, julkisen sektorin sähkö...  
391   [e-urheilu, vuorovaikutus, joukkueet, tietokon...  
992   [Energy Policy, Parliamentary Debates, Securit...  
987   [Language Learning, English as a Foreign Langu...  
To support model development and performance interpretation, a dataset-level analysis was conducted. While the primary goal of this work is to build a recommendation system that maps thesis abstracts to relevant tags, understanding the characteristics of the training data provides essential context for preprocessing and error analysis.
Abstract and Title Statistics

Thesis abstracts vary substantially in length, averaging 2,687 characters with a range from 90 to over 10,000. Titles are shorter but still diverse, with an average length of 95 characters and a maximum of 424. These variations influenced model input preparation, particularly with regard to truncation and tokenization strategies.
Faculty Representation and Imbalance

The dataset spans theses from nine faculty labels, although the university officially recognizes six. Some of the smaller faculty groups likely reflect legacy structures or subdivisions that have since merged into broader categories. For instance, Humanities (HUM) and Social Sciences (SOC) may now be subsumed under the broader Humanities and Social Sciences (HSS) label. As shown in Figure 1, there is a clear imbalance in thesis counts between faculties, with HSS contributing the largest volume of works.

Although the supervised models are trained on abstract-tag pairs, not faculty labels, such imbalances can indirectly affect tag distribution and topic coverage. Faculties with fewer theses may underrepresent certain domains or themes in the training set.
Temporal Trends and Structural Shifts

Figure 2 and Figure 3 illustrate the number of theses per faculty over time and the cumulative growth of the dataset. A sharp increase in data volume begins around 2005, with another peak near 2016. This may reflect changes in digital archiving policies or faculty restructuring, such as the emergence of EP (Education and Psychology) and HSS around 2017. These institutional changes may influence how themes and topics are categorized over time, but this was not explored in depth due to time constraints.
Dataset Limitations and Tag Distribution

The dataset includes only open-access Master’s theses available up to early 2024, which represents a subset of the university's total thesis output. Faculty names are not always consistent, and no ground-truth faculty-tag mappings exist. Nevertheless, an analysis of tag frequency reveals a long-tail distribution, indicating that while many tags appear only a few times, a core set dominates the dataset. This reflects a common pattern in real-world classification and tagging tasks and is considered during evaluation and model calibration.

Given the dataset’s linguistic and topical diversity, multilingual and generative modeling approaches were both explored. However, deeper correlations between faculty, time, and tag semantics remain outside the scope of this study and are only considered to the extent they may affect supervised learning outcomes.
✅ Dataset updated: data_split_v6

📊 Summary:
  Total entries: 8860
  Added language: 862
  Disagreed/fixed: 2008
  Skipped (empty abstract): 0

🧪 Disagreement breakdown:
  eng->fin: 1951 cases
  eng->swe: 9 cases
  fin->eng: 43 cases
  eng->sw: 1 cases
  eng->fr: 2 cases
  fin->swe: 2 cases
'''
