# AI-Inosuke Project

## 1. Data Collection & Preprocessing

The dataset was collected from multiple sources to replicate the **persona of Inosuke (Kimetsu no Yaiba)** in a natural way.  
All samples were normalized into the format:

```json
{"instruction": "...", "input": "...", "output": "..."}
```

### Dataset Statistics

| Source          | Samples |
|-----------------|---------|
| Persona         | 782     |
| Quotes          | 1,330   |
| Conversations   | 6,797   |
| Generic QA      | 350     |
| **Total**       | **9,259** |

**Dataset Distribution**

```mermaid
pie title Dataset Composition
  "Persona (8.4%)" : 782
  "Quotes (14.4%)" : 1330
  "Conversations (73.4%)" : 6797
  "Generic QA (3.8%)" : 350
```

- **Persona** → Defines Inosuke’s characteristics, personality, and style.  
- **Quotes** → Preserves original voice lines from the anime/manga.  
- **Conversations** → Multi-turn dialogues, ensuring natural back-and-forth interactions.  
- **Generic QA** → Covers common questions, boosting generalization.  

---

## 2. Qwen2.5 Instruct + QLoRA 4-bit

| Criterion | Explanation |
|-----------|-------------|
| **Base Model** | [Qwen2.5-3B Instruct](https://huggingface.co/Qwen) – lightweight yet powerful for dialogue tasks, with strong multilingual support including Vietnamese. |
| **Technique** | **QLoRA 4-bit** significantly reduces memory usage while maintaining performance close to full precision fine-tuning. |
| **Resources** | Optimized for **6GB VRAM GPUs (RTX 3060, etc.)**, making it feasible without high-end hardware. |
| **Efficiency** | Great balance between quality and compute efficiency, enabling persona training at scale. |

**QLoRA Workflow**

```mermaid
flowchart LR
    A[Base Model Qwen2.5-3B Instruct] --> B[Quantization (4-bit NF4)]
    B --> C[LoRA Fine-tuning Layers]
    C --> D[Inosuke Persona Fine-tuned Model]
```

---

## 3. Fine-tuning Process

- **Base model**: `Qwen2.5-3B-Instruct`  
- **Method**: QLoRA (4-bit quantization, LoRA adapters)  
- **Hardware**: RTX 3060 6GB VRAM  
- **Dataset size**: 9,259 samples  
- **Run time**: 12 hours 

---

## 4. Results 

Model successfully fine-tuned with limited GPU memory.  
Supports single-turn and short multi-turn dialogues.  
Persona-aligned responses consistent with Inosuke’s style.  

