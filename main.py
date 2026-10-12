from dotenv import load_dotenv

load_dotenv()  # Load credentials before importing pipeline modules.

from utils.audio_pocessor import process_input
from core.transcriber import transcribe_all
from core.summaize import summarize_chunk, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain

def run_pipeline(source: str, language: str="english")-> dict:
    print("starting AI Video Assistant")

    chunks = process_input(source)

    transcript = transcribe_all(chunks, language=language)
    print(f"raw transcription (first 300 characters){transcript[:300]}...")

    title = generate_title(transcript)

    summary= summarize_chunk(transcript)

    action_items = extract_action_items(transcript)

    decision = extract_key_decisions(transcript)
    questions = extract_questions(transcript)

    rag_chain = build_rag_chain(transcript)
    return {
        
        "transcript": transcript,
        "title": title,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decision,
        "open_questions": questions,
        "rag_chain": rag_chain
    }

if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"Title: {result['title']}")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nAction Items:\n{result['action_items']}")
    print(f"\nKey Decisions:\n{result['key_decisions']}")
    print(f"\nOpen Questions:\n{result['open_questions']}")
    print("=" * 60)

    # Phase 2 - Chat with your meeting via RAG
    print("\nChat with your meeting (type 'exit' to quit)")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            break
        if not question:
            continue
        response = rag_chain.invoke(question)
        print(f"\nAssistant: {response}\n")

 