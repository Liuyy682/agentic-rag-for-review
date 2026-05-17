import gradio as gr
from application.rag_application import RagApplication
import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

def create_gradio_ui():
    app = RagApplication.create()
    doc_manager = app.document_manager
    chat_interface = app.chat_interface
    
    def format_file_list():
        files = doc_manager.get_markdown_files()
        if not files:
            return "📭 No documents available in the knowledge base"
        return "\n".join([f"{f}" for f in files])

    def format_course_list():
        return doc_manager.get_course_list()

    def course_dropdown_update(value=None):
        choices = doc_manager.get_course_choices()
        selected = value if value in choices else None
        return gr.Dropdown(choices=choices, value=selected)
    
    def upload_handler(files, course_names, progress=gr.Progress()):
        if not files:
            return None, format_file_list(), format_course_list(), course_dropdown_update(), course_dropdown_update()
            
        summary = doc_manager.add_documents_detailed(
            files, 
            course_names=course_names,
            progress_callback=lambda p, desc: progress(p, desc=desc)
        )
        
        gr.Info(
            "✅ "
            f"Added: {summary.added} | "
            f"Skipped: {summary.skipped} | "
            f"Failed: {summary.failed} | "
            f"Courses updated: {summary.course_updated}"
        )
        return None, format_file_list(), format_course_list(), course_dropdown_update(), course_dropdown_update()
    
    def clear_handler():
        doc_manager.clear_all()
        gr.Info(f"🗑️ Removed all documents")
        return format_file_list(), format_course_list(), course_dropdown_update(), course_dropdown_update()

    def refresh_handler():
        return format_file_list(), format_course_list(), course_dropdown_update(), course_dropdown_update()

    def rename_course_handler(current_name, new_name):
        if not current_name or not new_name or not new_name.strip():
            gr.Warning("Select a course and enter a new course name.")
        elif doc_manager.rename_course(current_name, new_name):
            gr.Info("Course renamed")
        else:
            gr.Warning("Course rename skipped. Check the current and new names.")
        return format_course_list(), course_dropdown_update(new_name), course_dropdown_update(new_name)

    def rename_section_handler(course_name, current_section, new_section):
        if not course_name or not current_section or not new_section or not new_section.strip():
            gr.Warning("Select a course and enter the current and new section names.")
        elif doc_manager.rename_section(course_name, current_section, new_section):
            gr.Info("Section renamed")
        else:
            gr.Warning("Section rename skipped. Check the course and section names.")
        return format_course_list()
    
    def chat_handler(msg, hist, course_name):
        for chunk in chat_interface.chat(msg, hist, course_name):
            yield chunk
    
    def clear_chat_handler():
        chat_interface.clear_session()
    
    with gr.Blocks(title="Agentic RAG") as demo:
        
        with gr.Tab("Documents", elem_id="doc-management-tab"):
            gr.Markdown("## Add New Documents")
            gr.Markdown("Upload PDF, Markdown, Word, or PowerPoint files. Duplicates will be automatically skipped.")
            
            files_input = gr.File(
                label="Drop PDF, MD, DOCX, or PPTX files here",
                file_count="multiple",
                type="filepath",
                height=200,
                show_label=False
            )
            course_names_input = gr.Textbox(
                label="Course name(s)",
                placeholder="e.g. Database Systems, Computer Networks",
                lines=1
            )
            
            add_btn = gr.Button("Add Documents", variant="primary", size="md")
            
            gr.Markdown("## Current Documents in the Knowledge Base")
            file_list = gr.Textbox(
                value=format_file_list(),
                interactive=False,
                lines = 7,
                max_lines=10,
                elem_id="file-list-box",
                show_label=False
            )
            
            with gr.Row():
                refresh_btn = gr.Button("Refresh", size="md")
                clear_btn = gr.Button("Clear All", variant="stop", size="md")

            gr.Markdown("## Courses")
            course_list = gr.Textbox(
                value=format_course_list(),
                interactive=False,
                lines=6,
                max_lines=10,
                show_label=False
            )

            with gr.Row():
                edit_course_dropdown = gr.Dropdown(
                    label="Course",
                    choices=doc_manager.get_course_choices(),
                    interactive=True,
                )
                new_course_name = gr.Textbox(label="New course name", lines=1)
            rename_course_btn = gr.Button("Rename Course", size="md")

            with gr.Row():
                section_current_name = gr.Textbox(label="Current section name", lines=1)
                section_new_name = gr.Textbox(label="New section name", lines=1)
            rename_section_btn = gr.Button("Rename Section", size="md")
            
        with gr.Tab("Chat"):
            chat_course_dropdown = gr.Dropdown(
                label="Course scope",
                choices=doc_manager.get_course_choices(),
                interactive=True,
            )
            chatbot = gr.Chatbot(
                height=720, 
                placeholder="<strong>Ask me anything!</strong><br><em>I'll search, reason, and act to give you the best answer :)</em>",
                show_label=False,
                avatar_images=(None, os.path.join(ASSETS_DIR, "chatbot_avatar.png")),
                layout="bubble"
            )
            chatbot.clear(clear_chat_handler)
            
            gr.ChatInterface(fn=chat_handler, chatbot=chatbot, additional_inputs=[chat_course_dropdown])

        add_btn.click(
            upload_handler,
            [files_input, course_names_input],
            [files_input, file_list, course_list, edit_course_dropdown, chat_course_dropdown],
            show_progress="corner",
        )
        refresh_btn.click(refresh_handler, None, [file_list, course_list, edit_course_dropdown, chat_course_dropdown])
        clear_btn.click(clear_handler, None, [file_list, course_list, edit_course_dropdown, chat_course_dropdown])
        rename_course_btn.click(
            rename_course_handler,
            [edit_course_dropdown, new_course_name],
            [course_list, edit_course_dropdown, chat_course_dropdown],
        )
        rename_section_btn.click(
            rename_section_handler,
            [edit_course_dropdown, section_current_name, section_new_name],
            course_list,
        )
    
    return demo
