import streamlit as st
import os
from pathlib import Path
import json
import pandas as pd
from processors.document_processor import DocumentProcessor
from chunking.chunker import Chunker
from preprocessing.text_cleaner import TextCleaner
from vectorstores.vector_exporter import VectorExporter
from processors.document_structurer import DocumentStructurer

# Set page configuration
st.set_page_config(
    page_title="RAG Converter",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for compact design
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stMarkdown {
        font-size: 12px;
    }
    .stButton button {
        font-size: 12px;
        padding: 0.25rem 0.75rem;
    }
    .stSelectbox div div div {
        font-size: 12px;
    }
    .stTextArea textarea {
        font-size: 12px;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 12px;
    }
    .stProgress div {
        font-size: 12px;
    }
    div.stMarkdown p {
        font-size: 12px;
    }
    .streamlit-expanderHeader {
        font-size: 12px;
    }
    div[data-testid="stSidebarNav"] {
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_document' not in st.session_state:
    st.session_state.selected_document = None
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = {}
if 'preprocessing_options' not in st.session_state:
    st.session_state.preprocessing_options = {}
if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = False

# Initialize processors
doc_processor = DocumentProcessor()
chunker = Chunker()
text_cleaner = TextCleaner()
vector_exporter = VectorExporter()
doc_structurer = DocumentStructurer()

# Title and description
st.title("RAG Converter")
st.markdown("""
    Welcome to the RAG Converter! This tool helps you prepare your documents optimally 
    for RAG systems.
""")

def process_documents(files, chunk_strategy, chunk_size, overlap_percent, export_format):
    """Process uploaded documents and return chunks."""
    processed_docs = {}
    overlap = int(chunk_size * (overlap_percent / 100))
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, file in enumerate(files):
        try:
            # Update status
            status_text.text(f"Processing {file.name}...")
            
            # Read and process document
            content = file.read()
            doc_result = doc_processor.process_document(content, file.name)
            
            # Create chunks
            chunks = chunker.chunk_document(
                doc_result['content'],
                chunk_strategy,
                chunk_size,
                overlap
            )
            
            # Add metadata to chunks
            for chunk in chunks:
                chunk['metadata'] = doc_result['metadata']
            
            processed_docs[file.name] = {
                'content': doc_result['content'],
                'chunks': chunks,
                'metadata': doc_result['metadata']
            }
            
            # Update progress
            progress = (idx + 1) / len(files)
            progress_bar.progress(progress)
            
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    
    return processed_docs

def export_chunks(processed_docs, export_format):
    """Export processed chunks in the specified format."""
    if not processed_docs:
        return None
    
    all_chunks = []
    for doc_name, doc_data in processed_docs.items():
        all_chunks.extend(doc_data['chunks'])
    
    if export_format == "JSON":
        return json.dumps(all_chunks, indent=2, ensure_ascii=False)
    elif export_format == "JSONL":
        return '\n'.join(json.dumps(chunk, ensure_ascii=False) for chunk in all_chunks)
    elif export_format == "CSV":
        df = pd.DataFrame([{
            'text': chunk['text'],
            'start': chunk['start'],
            'end': chunk['end'],
            'strategy': chunk['strategy'],
            'filename': chunk['metadata']['filename'],
            'type': chunk['metadata']['type']
        } for chunk in all_chunks])
        return df.to_csv(index=False)
    else:  # Markdown
        md_content = []
        for chunk in all_chunks:
            md_content.append(f"## Chunk from {chunk['metadata']['filename']}\n")
            md_content.append(chunk['text'])
            md_content.append("\n---\n")
        return '\n'.join(md_content)

# Sidebar configuration
with st.sidebar:
    st.header("Settings")
    
    # Document upload
    uploaded_files = st.file_uploader(
        "Upload Documents",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'md', 'html', 'docx', 'xlsx', 'pptx'],
        help="""Upload your documents here. Supported formats:
        - PDF (with integrated OCR support for scanned documents)
        - Word (DOCX) - with table support
        - Excel (XLSX) - processes all worksheets
        - PowerPoint (PPTX) - text from all slides
        - Text, Markdown, HTML - with metadata extraction"""
    )
    
    # Smart Preprocessing Options
    st.subheader("Preprocessing")
    with st.expander("Preprocessing Options", expanded=False):
        preprocessing_options = {
            'fix_unicode': st.checkbox('Fix Unicode', True, help="Repairs faulty Unicode characters"),
            'remove_headers_footers': st.checkbox('Remove Headers/Footers', True, help="Removes repeating headers and footers"),
            'fix_line_breaks': st.checkbox('Fix Line Breaks', True, help="Corrects faulty line breaks and hyphenation"),
            'remove_urls': st.checkbox('Remove URLs', False, help="Removes URLs from the text"),
            'remove_emails': st.checkbox('Remove Emails', False, help="Removes email addresses from the text"),
            'remove_phone_numbers': st.checkbox('Remove Phone Numbers', False, help="Removes phone numbers from the text"),
            'remove_special_characters': st.checkbox('Remove Special Characters', False, help="Removes special characters and symbols"),
            'remove_empty_lines': st.checkbox('Remove Empty Lines', True, help="Removes unnecessary empty lines")
        }
        st.session_state.preprocessing_options = preprocessing_options
    
    # Document Structuring Options
    st.subheader("Document Structuring")
    with st.expander("Structuring Options", expanded=False):
        enable_structuring = st.checkbox(
            "Enable Smart Document Structuring",
            False,
            help="Automatically organize and structure documents based on content similarity and topics"
        )
        if enable_structuring:
            max_topics = st.slider(
                "Maximum Topics",
                min_value=2,
                max_value=10,
                value=5,
                help="Maximum number of topics to extract from documents"
            )
            min_similarity = st.slider(
                "Minimum Similarity",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                help="Minimum similarity threshold for grouping documents (higher = stricter grouping)"
            )
    
    # Chunking settings
    st.subheader("Chunking Settings")
    chunk_strategy = st.selectbox(
        "Chunking Strategy",
        ["Semantic", "Length-based", "Structure-based"],
        help="""Choose the method for splitting your documents:
        - Semantic: Splits text based on sentence structures and meaning
        - Length-based: Splits text into uniform pieces with intelligent breaks
        - Structure-based: Uses existing document structure (headings, paragraphs)"""
    )
    
    chunk_size = st.slider(
        "Chunk Size (characters)",
        min_value=100,
        max_value=2000,
        value=500,
        step=100,
        help="Target size for each chunk in characters. Larger chunks contain more context, smaller chunks are more precise for searching."
    )
    
    overlap = st.slider(
        "Overlap (%)",
        min_value=0,
        max_value=50,
        value=10,
        step=5,
        help="Percentage of overlap between chunks. More overlap helps preserve context between chunks."
    )
    
    # Export settings
    st.subheader("Export Settings")
    export_format = st.selectbox(
        "Export Format",
        ["JSON", "JSONL", "CSV", "Markdown", "Vector DB"],
        help="""Choose the export format:
        - JSON: Structured data with metadata
        - JSONL: Line-by-line JSON (good for large datasets)
        - CSV: Tabular format
        - Markdown: Formatted text
        - Vector DB: Direct export to vector database"""
    )
    
    if export_format == "Vector DB":
        vector_db = st.selectbox(
            "Vector Database",
            vector_exporter.get_supported_databases(),
            help="""Choose the target vector database:
            - Pinecone: Cloud-based, scalable
            - Weaviate: Self-hosted or cloud
            - Chroma: Local storage"""
        )
        
        if vector_db == "pinecone":
            api_key = st.text_input("Pinecone API Key", type="password")
            environment = st.text_input("Pinecone Environment")
            index_name = st.text_input("Index Name", value="rag-chunks")
        elif vector_db == "weaviate":
            weaviate_url = st.text_input("Weaviate URL")
            weaviate_key = st.text_input("API Key (optional)", type="password")
        elif vector_db == "chroma":
            persist_directory = st.text_input("Storage Directory", value="./chroma_db")
            collection_name = st.text_input("Collection Name", value="rag_chunks")

# Main content area
if uploaded_files:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Document Overview")
        for file in uploaded_files:
            if st.button(f"📄 {file.name}", key=file.name):
                st.session_state.selected_document = file.name
                st.session_state.show_analysis = True
    
    with col2:
        st.subheader("Preview")
        if st.session_state.selected_document and st.session_state.selected_document in st.session_state.processed_documents:
            doc = st.session_state.processed_documents[st.session_state.selected_document]
            
            # Show metadata and analysis
            with st.expander("Document Analysis", expanded=True):
                metadata = doc['metadata']
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Basic Information**")
                    st.json({
                        'Filename': metadata['filename'],
                        'Type': metadata['type'],
                        'Size': f"{metadata['size'] / 1024:.1f} KB",
                        'Language': metadata.get('language', 'unknown')
                    })
                
                with col2:
                    st.markdown("**Processing Statistics**")
                    if 'preprocessing_stats' in doc:
                        stats = doc['preprocessing_stats']
                        st.json({
                            'Characters reduced': f"{stats['reduction_percentage']}%",
                            'Original lines': stats['original_lines'],
                            'Cleaned lines': stats['cleaned_lines']
                        })
            
            # Show content
            st.markdown("**Original Text:**")
            st.text_area("", doc['content'], height=200)
            
            # Show chunks
            st.markdown("**Chunks:**")
            for i, chunk in enumerate(doc['chunks']):
                with st.expander(f"Chunk {i+1} ({len(chunk['text'])} characters)"):
                    st.text(chunk['text'])
                    st.caption(f"Position: {chunk['start']}-{chunk['end']}")
                    if 'metadata' in chunk:
                        st.json(chunk['metadata'])
        else:
            st.info("Select a document from the list to see a preview.")
    
    # Process button
    if st.button("Process Documents", type="primary"):
        with st.spinner("Processing documents..."):
            # Apply preprocessing options
            text_cleaner.set_options(st.session_state.preprocessing_options)
            
            processed_docs = {}
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Processing {file.name}...")
                    
                    # Read and process document
                    content = file.read()
                    doc_result = doc_processor.process_document(content, file.name)
                    
                    # Apply preprocessing
                    original_text = doc_result['content']
                    cleaned_text = text_cleaner.clean_text(original_text)
                    doc_result['content'] = cleaned_text
                    doc_result['preprocessing_stats'] = text_cleaner.get_preprocessing_stats(
                        original_text, cleaned_text
                    )
                    
                    # Create chunks
                    chunks = chunker.chunk_document(
                        cleaned_text,
                        chunk_strategy,
                        chunk_size,
                        overlap
                    )
                    
                    # Add metadata to chunks
                    for chunk in chunks:
                        chunk['metadata'] = doc_result['metadata']
                    
                    processed_docs[file.name] = {
                        'content': cleaned_text,
                        'chunks': chunks,
                        'metadata': doc_result['metadata'],
                        'preprocessing_stats': doc_result['preprocessing_stats']
                    }
                    
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
            
            progress_bar.empty()
            status_text.empty()
            
            st.session_state.processed_documents = processed_docs
            
            # Apply document structuring if enabled
            if enable_structuring and processed_docs:
                with st.spinner("Structuring documents..."):
                    structure_result = doc_structurer.structure_documents(
                        list(processed_docs.values()),
                        min_similarity=min_similarity,
                        max_topics=max_topics
                    )
                    
                    if structure_result['status'] == 'success':
                        structure = structure_result['structure']
                        
                        # Display structure information
                        st.subheader("Document Structure Analysis")
                        
                        # Display topics
                        with st.expander("Main Topics", expanded=True):
                            for i, topic_words in enumerate(structure['topics']):
                                st.markdown(f"**Topic {i+1}:** {', '.join(topic_words)}")
                        
                        # Display clusters
                        with st.expander("Document Clusters", expanded=True):
                            for cluster_id, cluster_info in structure['clusters'].items():
                                st.markdown(f"**{cluster_id}** (Size: {cluster_info['size']})")
                                st.markdown("Main topic: " + ', '.join(cluster_info['main_topic']))
                                st.markdown("Documents:")
                                for doc in cluster_info['documents']:
                                    st.markdown(f"- {doc['metadata']['filename']}")
                                st.markdown("---")
                        
                        # Display central documents
                        with st.expander("Key Documents", expanded=True):
                            st.markdown("Documents that are most connected to others:")
                            for doc in structure['central_documents']:
                                st.markdown(f"- {doc['metadata']['filename']}")
                        
                        # Display relationships
                        with st.expander("Document Relationships", expanded=True):
                            for rel in structure['relationships']:
                                st.markdown(
                                    f"- {rel['source']} → {rel['target']} "
                                    f"(Similarity: {rel['similarity']:.2f})"
                                )
                    else:
                        st.error(f"Error during document structuring: {structure_result['message']}")
            
            if processed_docs:
                st.success("Processing completed!")
                
                # Export results
                if export_format == "Vector DB":
                    all_chunks = []
                    for doc_data in processed_docs.values():
                        all_chunks.extend(doc_data['chunks'])
                    
                    try:
                        if vector_db == "pinecone":
                            result = vector_exporter.export_to_pinecone(
                                all_chunks, api_key, environment, index_name
                            )
                        elif vector_db == "weaviate":
                            result = vector_exporter.export_to_weaviate(
                                all_chunks, weaviate_url, weaviate_key
                            )
                        elif vector_db == "chroma":
                            result = vector_exporter.export_to_chroma(
                                all_chunks, persist_directory, collection_name
                            )
                        
                        if result['status'] == 'success':
                            st.success(f"Export to {vector_db} successful! {result['chunks_uploaded']} chunks exported.")
                        else:
                            st.error(f"Error exporting: {result['message']}")
                            
                    except Exception as e:
                        st.error(f"Error exporting to vector database: {str(e)}")
                
                else:
                    export_data = export_chunks(processed_docs, export_format)
                    if export_data:
                        file_extension = {
                            "JSON": "json",
                            "JSONL": "jsonl",
                            "CSV": "csv",
                            "Markdown": "md"
                        }[export_format]
                        
                        st.download_button(
                            label="Download Results",
                            data=export_data,
                            file_name=f"rag_chunks.{file_extension}",
                            mime={
                                "JSON": "application/json",
                                "JSONL": "application/jsonl",
                                "CSV": "text/csv",
                                "Markdown": "text/markdown"
                            }[export_format]
                        )
else:
    st.info("Please upload documents to start.") 