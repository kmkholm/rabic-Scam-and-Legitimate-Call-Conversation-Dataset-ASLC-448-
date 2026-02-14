# Arabic-Scam-and-Legitimate-Call-Conversation-Dataset-ASLC-448-

## Description

This dataset presents a novel multi-dialect Arabic scam and legitimate telephone call conversation corpus designed for training and evaluating scam detection models. The dataset addresses a critical gap in Arabic-language fraud detection research, where no publicly available scam call datasets currently exist.

The dataset contains 448 annotated conversations covering nine Arabic dialects: Modern Standard Arabic (MSA), Egyptian, Gulf, Jordanian, Saudi, Yemeni, Sudanese, Iraqi, and Syrian. Each conversation simulates a realistic telephone interaction structured as a multi-turn dialogue between a caller and a receiver over five utterance turns (three caller turns and two receiver turns).

The conversations are labeled as either scam (246 samples, 54.9%) or not_scam (202 samples, 45.1%) and span 23 fine-grained categories. The scam class includes 11 categories representing prevalent fraud tactics: bank fraud, prize/lottery scams, government impersonation, investment/crypto fraud, tech support scams, charity/emergency exploitation, romance/social engineering, fraudulent job offers, SIM/WhatsApp hijacking, rental/real estate fraud, and delivery/shipping scams. The non-scam class covers 12 categories of common legitimate calls: banking services, delivery notifications, utility services, personal/family calls, medical appointments, workplace communications, telecom offers, government services, restaurant orders, educational institutions, insurance services, and real estate inquiries.

Each conversation is additionally annotated with five numerical risk indicator scores that quantify the intensity of social engineering tactics: urgency score (0–5), sensitive information requests (0–2), financial pressure score (0–5), threat score (0–3), and impersonation score (0–2). These annotations provide granular insight into the manipulation strategies employed in scam calls and can serve as auxiliary features for classification models.

The audio component of the dataset consists of synthesized speech files generated from the text conversations using the Microsoft Edge Neural Text-to-Speech (TTS) engine. Dialect-specific voice pairs were assigned to each conversation (distinct male and female voices for caller and receiver), and post-processing was applied to simulate realistic telephone call conditions, including a bandpass filter (300–3400 Hz), light background noise, speed variation, and amplitude normalization. All audio files are in 16-bit PCM WAV format at 16 kHz mono, compatible with speech processing models such as OpenAI Whisper.

---

## Keywords

Arabic NLP; Scam Detection; Fraud Detection; Phone Call Classification; Multi-Dialect Arabic; Social Engineering; Text Classification; Speech Classification; Cybersecurity Dataset; AraBERT

---

## Files Included

| File | Description |
|------|-------------|
| arabic_scam_dataset_complete.xlsx | Complete text dataset with 448 conversations, labels, categories, dialects, risk scores, and metadata (18 columns) |
| audio_dataset/scam/*.wav | Synthesized audio files for scam conversations (16 kHz, mono, WAV) |
| audio_dataset/not_scam/*.wav | Synthesized audio files for legitimate conversations (16 kHz, mono, WAV) |

---

## Data Structure

The Excel file contains 18 columns per conversation:

| Column | Type | Description |
|--------|------|-------------|
| conversation_id | String | Unique identifier (CONV_0001 to CONV_0448) |
| full_conversation | String | Complete conversation text with speaker labels |
| caller_turn_1 | String | First caller utterance |
| receiver_turn_1 | String | First receiver response |
| caller_turn_2 | String | Second caller utterance |
| receiver_turn_2 | String | Second receiver response |
| caller_turn_3 | String | Third caller utterance |
| label | String | Binary class label: scam or not_scam |
| category | String | Fine-grained category (23 categories) |
| dialect | String | Arabic dialect (9 dialects) |
| urgency_score | Integer | Time pressure intensity (0–5) |
| sensitive_info_requests | Integer | Confidential data solicitation (0–2) |
| financial_pressure_score | Integer | Monetary demands intensity (0–5) |
| threat_score | Integer | Threat/intimidation level (0–3) |
| impersonation_score | Integer | Identity deception level (0–2) |
| conversation_length | Integer | Total characters in conversation |
| word_count | Integer | Total words in conversation |
| label_binary | Integer | Binary encoding: 1 = scam, 0 = not_scam |

---

## Categories

### Scam (11 categories, 246 conversations)
bank_fraud, prize_lottery, government_impersonation, investment_crypto, tech_support, charity_emergency, romance_social, job_offer, sim_whatsapp, rental_realestate, delivery_shipping

### Legitimate (12 categories, 202 conversations)
bank_legitimate, delivery_legitimate, service_legitimate, personal_family, medical_legitimate, work_legitimate, telecom_legitimate, government_legitimate, restaurant_legitimate, education_legitimate, insurance_legitimate, realestate_legitimate

---

## Suggested Uses

- Arabic text-based scam call classification using transformer models (e.g., AraBERT, CAMeLBERT)
- Audio-based scam detection using speech features (MFCC, mel-spectrograms)
- Multimodal scam detection combining text and audio modalities
- Cross-dialect Arabic NLP evaluation
- Social engineering tactic analysis and visualization
- Benchmarking Arabic language understanding models

---

## License

CC BY 4.0 (Creative Commons Attribution 4.0 International) To access to this dataset email me :kmkhol01@gmail.com
