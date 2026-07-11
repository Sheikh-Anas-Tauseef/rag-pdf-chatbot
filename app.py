"""Gradio app for the QA bot.

Lets the user upload one or more PDFs, then ask questions in a chat
that remembers earlier questions in the same session.
"""

import gradio as gr
from qa_bot import build_qa_pipeline

# Stores the QA chain built from the currently uploaded PDFs.
qa_chain_state = {"chain": None}


def process_pdfs(files):
    """Process one or more uploaded PDFs and build the QA chain."""

    if not files:
        return "No files uploaded yet.", []

    try:
        # Gradio gives a list of file objects, each with a temp path in .name.
        file_paths = [f.name for f in files]
        qa_chain_state["chain"] = build_qa_pipeline(file_paths)
        return f"{len(file_paths)} PDF(s) processed successfully. You can now ask questions.", []
    except Exception as e:
        qa_chain_state["chain"] = None
        return f"Error processing PDF(s): {e}", []


def answer_question(question, chat_history):
    """Send a question through the QA chain, using chat_history for context."""

    if qa_chain_state["chain"] is None:
        chat_history.append((question, "Please upload a PDF first."))
        return "", chat_history

    if not question or not question.strip():
        return "", chat_history

    try:
        # gr.Chatbot gives history as a list of lists, e.g. [[q, a], [q, a]],
        # but ConversationalRetrievalChain expects a list of tuples.
        history_as_tuples = [(pair[0], pair[1]) for pair in chat_history]

        response = qa_chain_state["chain"].invoke({
            "question": question,
            "chat_history": history_as_tuples,
        })
        answer = response["answer"]
    except Exception as e:
        answer = f"Error answering question: {e}"

    chat_history.append((question, answer))
    return "", chat_history


with gr.Blocks() as demo:
    gr.Markdown("# QA Bot — Ask Questions About Your PDFs")

    with gr.Row():
        pdf_upload = gr.File(label="Upload PDF(s)", file_types=[".pdf"], file_count="multiple")

    upload_status = gr.Textbox(label="Upload Status", interactive=False)

    chatbot = gr.Chatbot(label="Conversation")
    question_box = gr.Textbox(label="Ask a question about the PDF(s)")
    submit_btn = gr.Button("Submit")

    # Uploading new PDFs starts a fresh conversation, so we clear the chatbot too.
    pdf_upload.change(fn=process_pdfs, inputs=pdf_upload, outputs=[upload_status, chatbot])

    submit_btn.click(fn=answer_question, inputs=[question_box, chatbot], outputs=[question_box, chatbot])
    question_box.submit(fn=answer_question, inputs=[question_box, chatbot], outputs=[question_box, chatbot])


if __name__ == "__main__":
    demo.launch()
