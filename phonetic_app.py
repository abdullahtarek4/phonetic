import streamlit as st
import whisper
from g2p_en import G2p
import difflib
import os
import math
import nltk
from gtts import gTTS  
import io              

# ==========================================
# 0. INITIALIZE SESSION STATE
# ==========================================
st.set_page_config(page_title="MindBuzz Phonetic Prototype", page_icon="🧩")

if 'attempt_history' not in st.session_state:
    st.session_state.attempt_history = []
    
# --- NEW: Tracking failures for the TTS Hint ---
if 'fail_count' not in st.session_state:
    st.session_state.fail_count = 0
if 'current_target' not in st.session_state:
    st.session_state.current_target = "Snake"

# ==========================================
# 1. DOWNLOAD NLTK DATA & CACHE MODELS
# ==========================================
@st.cache_resource
def setup_nltk():
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True) 
    nltk.download('cmudict', quiet=True)
    nltk.download('punkt', quiet=True)

@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base.en")
    g2p_model = G2p() 
    return whisper_model, g2p_model

with st.spinner("⏳ Setting up NLTK dictionaries..."):
    setup_nltk()

with st.spinner("⏳ Loading Speech and Phoneme Models..."):
    whisper_model, g2p = load_models()

# ==========================================
# 2. UI HEADER
# ==========================================
st.title("🧩 MindBuzz: Phonetic Assessment")
st.write("Record a word. The AI will break it down into phonemes (sounds) to check your exact pronunciation.")

# ==========================================
# 3. USER INPUTS
# ==========================================
target_word = st.text_input("Target Word (e.g., Snake):", value=st.session_state.current_target)

# Reset the fail counter if the user types in a new word
if target_word.strip().lower() != st.session_state.current_target.strip().lower():
    st.session_state.current_target = target_word
    st.session_state.fail_count = 0

audio_file = st.audio_input("Record Audio Attempt")

# --- NEW: Display TTS Hint if they failed 3 times ---
if st.session_state.fail_count >= 3:
    st.info("💡 You've missed this a few times. Listen to how it is supposed to sound:")
    # Generate the audio in memory (no need to save a file)
    tts = gTTS(text=target_word, lang='en')
    sound_fp = io.BytesIO()
    tts.write_to_fp(sound_fp)
    sound_fp.seek(0)
    st.audio(sound_fp, format='audio/mp3')

if st.button("Evaluate Pronunciation", type="primary"):
    
    if audio_file is not None:
        temp_path = "temp_record.wav"
        with open(temp_path, "wb") as f:
            f.write(audio_file.getbuffer())
        
        with st.spinner("🤖 Extracting phonemes..."):
            vocab_prompt = f"{target_word}, take, make, bake, snack, snuck, fake."
            
            result = whisper_model.transcribe(
                temp_path, 
                fp16=False, 
                language="en",
                initial_prompt=vocab_prompt,
                condition_on_previous_text=False
            )
            
            recognized_word = result["text"].strip().lower().replace(".", "").replace("!", "").replace("?", "")
            
            confidence_score = 0.0
            if len(result["segments"]) > 0:
                logprob = result["segments"][0]["avg_logprob"]
                confidence_score = round(math.exp(logprob) * 100, 1)

            if "child will say" in recognized_word or recognized_word == "":
                recognized_word = "[Unclear]"
                
            # ==========================================
            # 4. PHONETIC EVALUATION LOGIC
            # ==========================================
            expected_clean = target_word.strip().lower()
            
            target_phonemes = [p for p in g2p(expected_clean) if p.isalnum()]
            recognized_phonemes = [p for p in g2p(recognized_word) if p.isalnum()] if recognized_word != "[Unclear]" else []
            
            phonetic_similarity = difflib.SequenceMatcher(None, target_phonemes, recognized_phonemes).ratio()
            
            st.divider()
            st.subheader("📊 Phonetic Breakdown")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Expected Word:**\n\n{target_word.capitalize()}\n\n🔊 `{' '.join(target_phonemes)}`")
            with col2:
                st.warning(f"**Heard Word:**\n\n{recognized_word.capitalize()}\n\n🔊 `{' '.join(recognized_phonemes)}`")
            with col3:
                score_color = "🟢" if phonetic_similarity > 0.8 else ("🟡" if phonetic_similarity > 0.4 else "🔴")
                st.success(f"**Phonetic Score:**\n\n{score_color} {int(phonetic_similarity * 100)}%")
            
            # ==========================================
            # 5. GRANULAR FEEDBACK & FAIL TRACKING
            # ==========================================
            is_success = False
            if phonetic_similarity == 1.0:
                feedback = "🎉 Perfect pronunciation! You hit every sound."
                st.success(feedback)
                st.balloons()
                is_success = True
                st.session_state.fail_count = 0  # Reset on success!
            elif recognized_word == "[Unclear]":
                feedback = "🔊 I didn't hear anything. Let's try saying it out loud!"
                st.error(feedback)
                st.session_state.fail_count += 1
            elif phonetic_similarity >= 0.5:
                missing_sounds = [p for p in target_phonemes if p not in recognized_phonemes]
                if missing_sounds:
                    feedback = f"💪 Close! But I missed the `{missing_sounds[0]}` sound. Let's try again!"
                else:
                    feedback = f"💪 Almost there! You said '{recognized_word}'. Keep practicing."
                st.warning(feedback)
                st.session_state.fail_count += 1
            else:
                feedback = f"🧠 Not quite! Listen closely to the sounds in '{target_word}' and try again."
                st.error(feedback)
                st.session_state.fail_count += 1
                
            st.session_state.attempt_history.append({
                "target": target_word,
                "recognized": recognized_word.capitalize(),
                "success": is_success,
                "score": int(phonetic_similarity * 100),
                "feedback": feedback
            })
                
        if os.path.exists(temp_path):
            os.remove(temp_path)
    else:
        st.error("Please record an audio attempt first using the microphone!")

# ==========================================
# 6. DISPLAY SESSION HISTORY
# ==========================================
if len(st.session_state.attempt_history) > 0:
    st.divider()
    st.subheader("📝 Session History")
    
    for i, attempt in enumerate(reversed(st.session_state.attempt_history)):
        attempt_num = len(st.session_state.attempt_history) - i
        icon = "✅" if attempt["success"] else "❌"
        with st.expander(f"Attempt {attempt_num}: {attempt['recognized']} ({attempt['score']}%) {icon}"):
            st.write(f"**Target:** {attempt['target']}")
            st.write(f"**Feedback Given:** {attempt['feedback']}")
            
    if st.button("Clear History"):
        st.session_state.attempt_history = []
        st.session_state.fail_count = 0
        st.rerun()
