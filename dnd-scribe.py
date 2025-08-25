import streamlit as st
import torch
from faster_whisper import WhisperModel
import os
import tempfile
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="D&D Session Transcriber",
    page_icon="🧙‍♂️",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .transcript-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        font-family: monospace;
        white-space: pre-wrap;
        max-height: 500px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def create_dnd_transcript(segments):
    """Create a clean D&D-style transcript without speaker diarization"""
    transcript_lines = []
    
    # Add header
    transcript_lines.append("D&D SESSION TRANSCRIPTION")
    transcript_lines.append("=" * 50)
    transcript_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    transcript_lines.append("")
    
    for segment in segments:
        # Add timestamp
        timestamp = format_timestamp(segment.start)
        transcript_lines.append(f"[{timestamp}]")
        
        # Add text content
        text = segment.text.strip()
        transcript_lines.append(text)
        transcript_lines.append("")
    
    return "\n".join(transcript_lines)

def get_model_size():
    """Recommend model size based on your use case"""
    st.markdown("### Model Selection")
    st.info("For D&D sessions (multi-hour), medium-large models work well.")
    st.info("Larger models = better accuracy but slower processing")
    
    model_options = {
        "Medium": "medium",
        "Large-v2": "large-v2", 
        "Large-v3": "large-v3"
    }
    
    selected_model = st.radio(
        "Choose model size:",
        options=list(model_options.keys()),
        index=1,  # Default to large-v2
        help="Medium for balance, Large-v2/Large-v3 for better accuracy"
    )
    
    return model_options[selected_model]

def setup_device_selection(cuda_available):
    """Setup device selection with manual override option"""
    st.markdown("### Device Selection")
    st.info("Choose processing device. AUTO will automatically detect GPU availability.")
    
    # Create radio buttons for device selection
    device_options = {
        "AUTO": "auto",
        "GPU": "cuda",
        "CPU": "cpu"
    }
    
    selected_device = st.radio(
        "Select processing device:",
        options=list(device_options.keys()),
        index=0 if cuda_available else 2,  # Default to AUTO if CUDA available, CPU otherwise
        help="AUTO automatically selects GPU if available, otherwise uses CPU"
    )
    
    # Show device status
    if selected_device == "AUTO":
        device = "cuda" if cuda_available else "cpu"
        st.info(f"Auto-selecting: {'GPU' if cuda_available else 'CPU'}")
    else:
        device = device_options[selected_device]
        st.info(f"Manually selecting: {selected_device}")
    
    return device

def check_cuda_availability():
    """Check if CUDA is available and provide helpful messages"""
    if torch.cuda.is_available():
        st.success(f"✅ CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        return True
    else:
        st.warning("⚠️ CUDA is not available. Will use CPU (slower but functional)")
        st.info("To enable GPU acceleration, ensure you have:")
        st.markdown("""
        1. NVIDIA GPU with CUDA support
        2. Latest NVIDIA drivers installed
        3. CUDA Toolkit 11.8 or later
        """)
        return False

def download_transcript_to_downloads(transcript_content, filename):
    """Download transcript to user's downloads folder"""
    try:
        # Get user's downloads directory
        import os
        downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        
        # Create full file path
        file_path = os.path.join(downloads_dir, filename)
        
        # Write the transcript to the downloads folder
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(transcript_content)
        
        st.success(f"✅ Transcript saved to: {file_path}")
        return True
    except Exception as e:
        st.error(f"❌ Failed to save transcript to downloads: {str(e)}")
        return False

def main():
    st.title("🧙‍♂️ D&D Session Transcriber")
    st.markdown("""
    Upload your D&D session audio files and get clean, structured transcripts.
    Perfect for keeping session history and campaign notes!
    """)
    
    # Check CUDA availability
    cuda_available = check_cuda_availability()
    
    # Model selection
    model_size = get_model_size()
    
    # Device selection
    device = setup_device_selection(cuda_available)
    
    # File upload
    st.markdown("### Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose your audio file (MP3, WAV, M4A, FLAC)", 
        type=["mp3", "wav", "m4a", "flac"],
        help="Large files may take several minutes to process"
    )
    
    # Initialize session state for transcript
    if 'transcript_generated' not in st.session_state:
        st.session_state.transcript_generated = False
        st.session_state.transcript_content = ""
        st.session_state.transcript_filename = ""
    
    if uploaded_file is not None:
        # Display file info
        st.info(f"File: {uploaded_file.name}")
        st.info(f"Size: {uploaded_file.size / (1024*1024):.2f} MB")
        
        # Process button
        if st.button("🚀 Transcribe Session", type="primary"):
            with st.spinner("Processing audio file... This may take several minutes."):
                try:
                    # Create temporary file
                    temp_dir = tempfile.mkdtemp()
                    temp_file_path = os.path.join(temp_dir, "temp_audio.wav")
                    
                    # Save uploaded file
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load model with proper device handling
                    st.info("Loading Whisper model...")
                    
                    # Determine compute type based on device
                    if device == "cuda":
                        compute_type = "float16"
                    else:
                        compute_type = "int8"
                    
                    # Try to load model with specified device
                    try:
                        model = WhisperModel(
                            model_size, 
                            device=device,
                            compute_type=compute_type
                        )
                        
                        st.success("✅ Model loaded successfully!")
                        
                    except Exception as e:
                        st.warning(f"⚠️ Device loading failed: {str(e)}")
                        # Fallback to CPU if specified device fails
                        st.info("Falling back to CPU processing...")
                        model = WhisperModel(
                            model_size, 
                            device="cpu",
                            compute_type="int8"
                        )
                        st.success("✅ Model loaded on CPU!")
                    
                    # Transcribe with timestamps
                    st.info("Transcribing audio...")
                    segments, info = model.transcribe(
                        temp_file_path,
                        beam_size=5,
                        word_timestamps=True,
                        condition_on_previous_text=False,
                        vad_filter=True  # Voice activity detection
                    )
                    
                    # Convert segments to list for processing
                    segment_list = list(segments)
                    
                    # Create formatted transcript
                    st.success("Transcription complete!")
                    formatted_transcript = create_dnd_transcript(segment_list)
                    
                    # Store in session state
                    st.session_state.transcript_content = formatted_transcript
                    st.session_state.transcript_filename = f"dnd_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    st.session_state.transcript_generated = True
                    
                    # Show model info
                    st.info(f"Segments processed: {len(segment_list)}")
                    st.info(f"Total time: {format_timestamp(info.duration)}")
                    
                    # Automatically download to downloads folder
                    with st.spinner("Saving transcript to your downloads folder..."):
                        success = download_transcript_to_downloads(
                            formatted_transcript, 
                            st.session_state.transcript_filename
                        )
                        if success:
                            st.success("✅ Transcript automatically saved to your Downloads folder!")
                        else:
                            st.warning("⚠️ Could not save to downloads folder. Please use the download button below.")
                    
                except Exception as e:
                    st.error(f"Error during transcription: {str(e)}")
                    st.info("Make sure your audio file is not corrupted and try again.")
                    st.info("If the error persists, try using a smaller model size or check CUDA installation.")
                
                finally:
                    # Clean up temporary files
                    try:
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)
                        os.rmdir(temp_dir)  # This will fail if directory isn't empty, but that's okay
                    except:
                        pass

if __name__ == "__main__":
    main()
