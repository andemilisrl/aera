# ÆRA-4B

<div align="center">

[🚀 **Try Demo**](https://aera.andemili.com/) | [🤗 **Hugging Face**](https://huggingface.co/and-emili/aera-4b)

</div>

## Overview

ÆRA is a specialized 4 billion parameter language model developed by [AND EMILI](https://www.andemili.com/) as an enterprise-focused foundation for building intelligent agents and automation pipelines. Unlike general-purpose conversational models, ÆRA is intentionally designed with a narrow, practical focus on context-based reasoning and structured outputs.

## 🤗 Model Weights & Download

**Model weights are available on Hugging Face**: [https://huggingface.co/and-emili/aera-4b](https://huggingface.co/and-emili/aera-4b)

Download or explore the model directly from Hugging Face. This GitHub repository contains code examples, inference setups, and deployment guides for running ÆRA-4B.

## Key Capabilities

### 🇮🇹 Native Italian Language Support
ÆRA excels at understanding and generating Italian text, making it ideal for Italian-speaking enterprises and applications.

### 📄 Context-Only Responses
ÆRA is trained to rely exclusively on provided context rather than internal knowledge. When asked questions without relevant context, it will respond honestly:

> "Currently I don't have access to information about the actors who played Dr. Who. Feel free to share content and I will analyze it and tell you what I can infer from it."

This behavior ensures reliability and reduces hallucination in enterprise applications.

### 🔧 Structured Output Generation
- **JSON Generation**: Reliably produces well-formed JSON outputs
- **Entity Extraction**: Identifies and extracts entities from provided text
- **Classification**: Categorizes content based on given criteria
- **Sentiment Analysis**: Analyzes emotional tone in context

### 🛠️ Function Calling
Native support for tool use and function calling, enabling seamless integration into agentic workflows and automation pipelines.

## Design Philosophy

ÆRA is not intended to be a general-knowledge assistant like ChatGPT. Instead, it serves as a lightweight, efficient starting point for enterprises exploring:

- **Retrieval Augmented Generation (RAG)** implementations
- **Document analysis** and information extraction
- **Automated workflows** with structured outputs
- **Multi-agent systems** requiring reliable, predictable behavior

## Use Cases

This model is ideal for companies looking to:
- Test the viability of RAG systems for their specific needs
- Build proof-of-concepts for document processing pipelines
- Implement lightweight automation without cloud dependencies
- Evaluate whether LLM-based solutions fit their requirements

If initial tests with ÆRA prove successful, organizations can then invest in developing more specialized, powerful models tailored to their specific domain needs.

## Technical Details

- **Parameters**: 4 billion
- **Training**: Post-trained on synthetic data focused on structured reasoning and Italian language tasks
- **Deployment**: Optimized for local deployment on standard hardware
- **Privacy**: Runs entirely on-premises with no external API calls

## Getting Started

### Using Pipeline (Simplest)
```python
from transformers import pipeline

pipe = pipeline("text-generation", model="and-emili/aera-4b")
messages = [
    {"role": "user", "content": "Chi sei?"},
]
answer = pipe(messages)[0]['generated_text'][-1]['content']

print(answer) 
# Output: 'Ciao! Mi chiamo ÆRA, un assistente virtuale sviluppato da AND EMILI.'
```

### Direct Model Loading
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("and-emili/aera-4b")
model = AutoModelForCausalLM.from_pretrained("and-emili/aera-4b")

messages = [
    {"role": "user", "content": "Chi è L'attuale presidente della Repubblica Italiana?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=400)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
# Output: 'Al momento non ho informazioni aggiornate sull'attuale presidente della Repubblica Italiana. 
#         Se hai un testo o dei dati specifici che vuoi condividere, posso aiutarti a estrarre questa informazione.'
```

### RAG-Style Context Analysis
```python
from transformers import pipeline

pipe = pipeline("text-generation", model="and-emili/aera-4b")

# Document/context
document = """
Il nuovo prodotto XYZ-3000 è stato lanciato nel 2024 con un prezzo di €1,299. 
Include 3 anni di garanzia e supporto tecnico gratuito. Il prodotto pesa 2.5kg 
ed è disponibile in tre colori: nero, argento e blu. La batteria dura 48 ore 
con uso normale.
"""

messages = [
    {"role": "system", "content": document},
    {"role": "user", "content": "Quanto costa il prodotto e quali colori sono disponibili?"}
]

response = pipe(messages, max_new_tokens=100, temperature=0.3)[0]['generated_text'][-1]['content']
print(response) 
# Output: "Il prodotto XYZ-3000 costa €1,299 e è disponibile in tre colori: nero, argento e blu."
```

## OpenAI-Compatible API (via VLLM)

For production deployments, ÆRA supports OpenAI-compatible endpoints through VLLM, enabling structured output with Pydantic schemas:

```python

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, List

client = OpenAI(
    api_key="your-key",
    base_url="https://your-vllm-endpoint/v1",
)

# Complex structured output for meeting analysis
class ActionItem(BaseModel):
    azione: str = Field(description="Descrizione dell'azione da intraprendere")
    responsabile: Optional[str] = Field(description="Persona responsabile")
    scadenza: Optional[str] = Field(description="Data di scadenza")
    priorita: str = Field(description="Priorità: alta, media, bassa")

class MeetingSummary(BaseModel):
    riassunto: str = Field(description="Riassunto generale della riunione")
    decisioni_prese: List[str] = Field(description="Lista delle decisioni prese")
    azioni_da_intraprendere: List[ActionItem] = Field(description="Azioni specifiche da intraprendere")
    partecipanti: List[str] = Field(default=[], description="Lista dei partecipanti")
    prossima_riunione: Optional[str] = Field(description="Data della prossima riunione se menzionata")

# Real meeting notes to analyze
meeting_notes = """
Riunione del 15 giugno 2024 - Team Marketing
Presenti: Laura Bianchi (Marketing Manager), Marco Verdi (Social Media), Sara Neri (Grafica)

Discusso nuovo piano marketing Q3:
- Approvato budget €15.000 per campagna social media
- Laura coordinerà con agenzia esterna per video promozionali
- Marco deve preparare content calendar entro 30 giugno
- Sara creerà mockup nuova brochure entro 25 giugno
- Decidere fornitori stampa entro luglio
- Prossimo meeting: 29 giugno ore 14:00

Priorità alta: lancio campagna entro 15 luglio
Marco deve anche analizzare performance attuali social
"""

completion = client.beta.chat.completions.parse(
    model="and-emili/aera-4b",
    messages=[
        {"role": "system", "content": "Sei un assistente esperto che riassume riunioni aziendali italiane."},
        {"role": "user", "content": f"Analizza e riassumi questi appunti:\n\n{meeting_notes}"}
    ],
    response_format=MeetingSummary,
    temperature=0.5
)

result = completion.choices[0].message.parsed
print(f"RIASSUNTO: {result.riassunto}\n")
print(f"DECISIONI PRESE: {', '.join(result.decisioni_prese)}\n")
print("AZIONI DA INTRAPRENDERE:")
for action in result.azioni_da_intraprendere:
    print(f"- {action.azione}")
    if action.responsabile:
        print(f"  Responsabile: {action.responsabile}")
    print(f"  Priorità: {action.priorita}")



# Customer Support Automation with Escalation Logic
class CustomerResponse(BaseModel):
    risposta: str = Field(description="Risposta professionale al cliente")
    categoria_richiesta: str = Field(description="Categoria: spedizione, reso, pagamento, etc.")
    livello_urgenza: str = Field(description="Urgenza: basso, medio, alto")
    azioni_suggerite: List[str] = Field(description="Azioni che il cliente può intraprendere")
    escalation_richiesta: bool = Field(description="Se necessita escalation a operatore umano")

inquiry = "URGENTE! Il mio ordine per il matrimonio di domani non è ancora arrivato! Avevo pagato la spedizione express!"

completion = client.beta.chat.completions.parse(
    model="and-emili/aera-4b",
    messages=[
        {"role": "system", "content": "Sei un assistente clienti professionale per e-commerce."},
        {"role": "user", "content": inquiry}
    ],
    response_format=CustomerResponse,
    temperature=0.5
)

response = completion.choices[0].message.parsed
print(f"Urgenza: {response.livello_urgenza}")        # "alto"
print(f"Escalation: {response.escalation_richiesta}") # True
print(f"Risposta: {response.risposta}")
```

### Starter example in this repo

A working starter example to run ÆRA through an OpenAI-compatible API with vLLM on Modal is included under `serve_modal/`. It exposes `/v1/chat/completions` and `/health` and supports tool calling via a custom parser.

What's included:
- `serve_modal/README.txt`: deployment steps (Modal tokens, deploy command)
- `serve_modal/vllm_aera4b_inference.py`: Modal app that serves `and-emili/aera-4b` with OpenAI-compatible endpoints and a local test entrypoint
- `serve_modal/toolcall_parser.py`: vLLM tool-call parser for ÆRA's `<tool_call>` format (streaming and non-streaming)

Quick start:
```bash
cd serve_modal
pip install -U modal
export MODAL_TOKEN_ID="your_token_id"
export MODAL_TOKEN_SECRET="your_token_secret"
modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"
modal deploy vllm_aera4b_inference.py
```

Then point your OpenAI SDK to the deployed base URL and use the `Authorization: Bearer` header with your API key. See `serve_modal/README.txt` for details.

## Limitations

- Does not provide information beyond what's in the given context
- Not suitable for open-ended creative tasks or general knowledge queries
- Optimized for Italian; performance may vary in other languages
- Designed for specific enterprise use cases, not general conversation

## About AND EMILI

[AND EMILI](https://www.andemili.com/) specializes in developing practical AI solutions for enterprise automation and intelligence augmentation.

---

**License**: Apache 2.0
