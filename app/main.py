import re
from dotenv import load_dotenv
import speech_recognition as sr
from langgraph.checkpoint.mongodb import MongoDBSaver
load_dotenv()
from agent.graph import delete_path
from agent.graph import create_chat_graph

MONGODB_URI = "mongodb://admin:admin@localhost:27018/?authSource=admin"
config = {"configurable":{"thread_id":"2"}}


def extract_filename(user_text):
    words = user_text.lower().split()

    # remove obvious command words
    ignore_words = {"delete", "remove", "file", "please", "can", "you", "the", "a", "an"}

    # filter meaningful words
    candidates = [w for w in words if w not in ignore_words]

    if not candidates:
        return None

    # take the most likely filename (last meaningful word)
    name = candidates[-1]

    if "." not in name:
        name += ".py"

    return f"chat_gpt/{name}"


def main():
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph = create_chat_graph(checkpointer=checkpointer)

        r = sr.Recognizer()
        exit_words = ["exit", "quit", "stop", "shutdown", "close"]
        negation_words = ["not", "don't", "dont", "do not", "no"]
        last_message_id = None
        pending_delete = None
        while True:
            try:

                with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source)
                    print("Listening... Speak now.")
                    try:
                        audio = r.listen(source, timeout=5, phrase_time_limit=10)
                    except sr.WaitTimeoutError:
                        print("No speech detected.") 
                        continue   
                sst = r.recognize_google(audio)
                print(f"You said: {sst}")
                # Exit condition
                user_text = sst.lower().strip()
                   
                if pending_delete:
                    if any(word in user_text for word in ["yes", "confirm", "sure", "ok"]):
                        result = delete_path.invoke({"path": pending_delete})
                        print(result)
                        pending_delete = None
                        continue

                    else:
                        print("Deletion cancelled.")
                        pending_delete = None
                        continue       

                elif "delete" in user_text or "remove" in user_text:
                    target = extract_filename(user_text)

                    if not target:
                        print("Please specify the file to delete.")
                        continue

                    pending_delete = target

                    print(f"Are you sure you want to delete {target}? Please confirm.")
                    continue



                if any(n in user_text for n in negation_words) and any(w in user_text for w in exit_words):
                    break


                elif any(w in user_text for w in exit_words):
                    print("Exiting Voice Cursor...")
                    break


                for event in graph.stream({"messages":[{"role":"user","content":sst}]},config,stream_mode="values"):
                    if "messages" in event:
                        msg = event["messages"][-1]
                        if id(msg) == last_message_id:
                            continue
                        last_message_id = id(msg)


                        if msg.type == "ai" and getattr(msg, "tool_calls", None):
                            tool = msg.tool_calls[0]
                            print(f"\nTool Call → {tool['name']}")


                        elif msg.type == "ai" and msg.content and not msg.content.strip().startswith("{"):
                            print(msg.content)

            except sr.UnknownValueError:
                print("Could not understand audio. Please try again.")

            except sr.RequestError as e:
                print(f"Speech recognition error: {e}")

            except KeyboardInterrupt:
                print("\nStopping Voice Cursor...")
                break            

if __name__ == "__main__":
    main()       