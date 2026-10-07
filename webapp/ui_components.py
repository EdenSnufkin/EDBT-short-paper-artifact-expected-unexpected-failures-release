# webapp/ui_components.py
"""Streamlit UI building blocks: page text, question rendering, and the 7-point NASA-TLX-style Likert widget."""
import json
import time

import streamlit as st

from latex_rendering import split_latex_segments
from questions import ContentQuestion, answer_display_text, is_correct_answer
from survey import SCALE_MAX_LABEL, SCALE_MIN_LABEL, SCALE_POINTS


# -------------------------
# Global header / styling
# -------------------------

def header(page_key: str):
    """Render the stable app header and scroll when the logical page changes.

    This stays as the first element on every run. Keeping one permanent element
    prevents Streamlit from reconciling later content into a removed scroll-helper slot.
    """
    header_html = """
        <style>
        div[data-testid="stRadio"] > div[role="radiogroup"]{
            display: flex;
            flex-direction: row;
            gap: 0.5rem;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"]{
            flex: 1 1 0;
            margin: 0 !important;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child{
            display: none !important;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"] input + div{
            width: 100%;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 10px;
            padding: 0.55rem 0;
            justify-content: center;
            text-align: center;
            cursor: pointer;
        }
        div[data-testid="stRadio"] label[data-baseweb="radio"] input:checked + div{
            background: #ff4b4b !important;
            color: white !important;
            border-color: #ff4b4b !important;
            font-weight: 700;
        }
        </style>
        <script>
        (() => {
            const pageKey = __PAGE_KEY__;
            if ('scrollRestoration' in window.history) {
                window.history.scrollRestoration = 'manual';
            }
            if (window.__studyPageKey === pageKey) {
                return;
            }
            window.__studyPageKey = pageKey;

            const scrollToTop = () => {
                const candidates = [
                    document.scrollingElement,
                    document.documentElement,
                    document.body,
                    document.querySelector('[data-testid="stMain"]'),
                    document.querySelector('[data-testid="stAppViewContainer"]'),
                    document.querySelector('.stApp'),
                ];
                for (const candidate of candidates) {
                    if (!candidate) continue;
                    candidate.scrollTop = 0;
                    if (typeof candidate.scrollTo === 'function') {
                        candidate.scrollTo(0, 0);
                    }
                }
                window.scrollTo(0, 0);
            };

            window.requestAnimationFrame(scrollToTop);
            window.setTimeout(scrollToTop, 50);
            window.setTimeout(scrollToTop, 200);
        })();
        </script>
        <div style="text-align:center; margin-bottom:16px;">
            <h3 style="margin:0;">Adaptive Question Recommendation Study</h3>
        </div>
        """
    st.html(
        header_html.replace("__PAGE_KEY__", json.dumps(page_key)),
        unsafe_allow_javascript=True,
    )


# -------------------------
# Phase text blocks
# -------------------------

def landing_overview() -> bool:
    st.markdown(
        """
Welcome! In this study, you will answer a series of questions recommended by an adaptive
system, and share brief feedback about your experience.

#### Study structure
1. **Preliminary test:** a short set of questions covering a range of difficulty levels.
2. **Question batches:** the system recommends further questions as you progress.
3. **Short surveys:** a few brief check-ins about your experience.

Click **Next** to begin.
"""
    )
    st.markdown(
        """
        <p><span style="color:#b91c1c; font-weight:700;">Note: </span>
        Participation is voluntary. You may stop at any time using the
        <strong>End session</strong> button.</p>
        """,
        unsafe_allow_html=True,
    )
    return st.button("Next")


def consent_block() -> bool:
    st.markdown(
        """
<h3>Consent and data use</h3>
<p>This study is conducted for <strong>research purposes</strong> to study how people interact
with a system that recommends questions of adaptive difficulty.</p>
<p><strong>During the session, we will record:</strong></p>
<ul>
    <li>your answers (correct or incorrect) and response times</li>
    <li>the questions shown and the system's recommendations</li>
    <li>your survey responses</li>
</ul>
<p>No personally identifying information is collected beyond your Prolific ID, which is used
only to manage participation and payment.</p>
<p>Your participation is voluntary, and you may stop at any time.</p>
<p>Please confirm below if you agree to participate.</p>
""",
        unsafe_allow_html=True,
    )
    return st.checkbox("I consent to participate.", value=False)


def pretest_intro(n_questions: int) -> bool:
    st.markdown(
        f"""
## Preliminary test ({n_questions} questions)

You will answer **{n_questions} questions** spanning a range of difficulty levels.
This helps the system calibrate the questions shown to you later.

Please answer each question to the best of your ability; take your time.
"""
    )
    return st.button("Start preliminary test")


def pretest_results_block(score: int, total: int) -> bool:
    st.markdown("## Preliminary test results")
    st.write(f"**Score:** {score}/{total}")
    st.success("Thank you! You can continue to the next phase of the study.")
    return st.button("Continue to question batches")


def learning_intro(n_batches: int, batch_size: int) -> bool:
    st.markdown(
        f"""
## Question batches

You will work through **up to {n_batches} batches** of **{batch_size} questions each**.
The system selects each batch based on your previous answers.

You may be asked short survey questions between batches; please answer them as
carefully as you can. You may stop at any time using **End session**.
"""
    )
    return st.button("Start question batches")


def learning_done_block() -> bool:
    st.success("Thank you for completing the question batches!")
    return st.button("Continue")


def survey_intro():
    """Heading shown above the first survey administration (items follow immediately below)."""
    st.markdown(
        f"""
## Your feedback

Please answer each item on a **{SCALE_POINTS}-point scale**, from **{SCALE_MIN_LABEL}**
to **{SCALE_MAX_LABEL}**.
"""
    )


def _prolific_redirect(completion_code: str, completion_url: str) -> None:
    """Redirect to Prolific with a manual fallback; without a completion code, nothing more is shown."""
    if not completion_code:
        return
    st.markdown(
        f"""
You should be redirected to Prolific automatically. If the redirect does not work,
[click here to complete the study on Prolific]({completion_url}) or copy the code below.
"""
    )
    st.markdown("### Completion code")
    st.code(completion_code, language=None)
    st.html(
        """
        <script>
        window.setTimeout(() => {
            window.top.location.replace(__COMPLETION_URL__);
        }, 750);
        </script>
        """.replace("__COMPLETION_URL__", json.dumps(completion_url)),
        unsafe_allow_javascript=True,
    )


def study_complete_block(completion_code: str, completion_url: str):
    st.markdown(
        """
## Thank you!

Thank you for participating in this research study. Your responses help us evaluate
and improve adaptive question-recommendation methods.
"""
    )
    _prolific_redirect(completion_code, completion_url)


def study_ended_block(completion_code: str, completion_url: str):
    st.markdown(
        """
## Session ended

You have chosen to end the session. This completion code records the submission
without a bonus payment.
"""
    )
    _prolific_redirect(completion_code, completion_url)


def prolific_entry_error_page():
    st.error("Invalid study entry")
    st.markdown(
        """
This study must be accessed **directly from Prolific**.

Please return to Prolific and open the study again using the **Start Study** button.
If you believe this is an error, please contact the researcher via Prolific.

*(Local testing: set `STUDY_ALLOW_DEMO=1` or add `?PROLIFIC_PID=<any id>` to the URL.)*
"""
    )
    st.stop()


# -------------------------
# LaTeX-aware question text rendering (the questions use LaTeX)
# -------------------------

def _render_latex_text(text: str) -> None:
    for segment in split_latex_segments(text):
        if segment.kind == "latex":
            st.latex(segment.content)
        else:
            st.markdown(segment.content)


# -------------------------
# Question rendering
# -------------------------

def render_question(content: ContentQuestion, key_prefix: str) -> str | None:
    """Render one question and return the participant's raw chosen answer
    (option index as a string for MCQ, free text otherwise), or None if unanswered."""
    _render_latex_text(content.text)

    if content.options is not None:
        choice_key = f"{key_prefix}_choice_idx"
        if choice_key not in st.session_state:
            st.session_state[choice_key] = -1
        chosen_idx = int(st.session_state[choice_key])
        letters = [chr(ord("A") + i) for i in range(len(content.options))]

        for i, option_text in enumerate(content.options):
            left, right = st.columns([1, 12])
            btn_type = "primary" if chosen_idx == i else "secondary"
            clicked = left.button(
                letters[i], key=f"{key_prefix}_pick_{i}", type=btn_type, use_container_width=True
            )
            with right:
                _render_latex_text(option_text)
            if clicked:
                st.session_state[choice_key] = i
                st.rerun()

        return None if chosen_idx < 0 else str(chosen_idx)

    text_key = f"{key_prefix}_free_text"
    value = st.text_input("Your answer", key=text_key, label_visibility="collapsed")
    return value if value.strip() else None


def _render_review_answer(label: str, answer: str, key: str, color: str, background: str) -> None:
    """Render one color-coded answer while preserving the existing LaTeX handling."""
    with st.container(key=key, border=True):
        st.markdown(f":{color}[**{label}**]")
        _render_latex_text(answer)
    st.markdown(
        f"<style>.st-key-{key} {{"
        f"background-color: {background};"
        f"border-color: {color} !important;"
        "border-radius: 0.5rem;"
        "padding: 0.75rem 1rem;"
        "}}</style>",
        unsafe_allow_html=True,
    )


def render_answer_review(
    content: ContentQuestion,
    chosen_answer: str,
    key_prefix: str,
) -> None:
    """Show the correct answer in green and an incorrect learner answer in red."""
    _render_latex_text(content.text)

    if content.options is not None:
        correct_answer = str(content.correct_answer_index)
    else:
        correct_answer = content.correct_answer_text or ""

    correct = is_correct_answer(content, chosen_answer)
    correct_label = "Correct answer (your answer)" if correct else "Correct answer"
    _render_review_answer(
        correct_label,
        answer_display_text(content, correct_answer),
        key=f"{key_prefix}_correct",
        color="green",
        background="rgba(22, 163, 74, 0.10)",
    )

    if not correct:
        _render_review_answer(
            "Your answer",
            answer_display_text(content, chosen_answer),
            key=f"{key_prefix}_incorrect",
            color="red",
            background="rgba(220, 38, 38, 0.10)",
        )


# -------------------------
# 7-point NASA-TLX-style Likert widget
# -------------------------

def likert7(prompt: str, key: str, previous: int | None = None) -> int | None:
    """Low -> High, 1..SCALE_POINTS. Pre-filled with `previous` when given. If the
    participant then picks a different value, the pill matching their previous
    answer keeps a distinct highlight so that answer stays visible for comparison."""
    st.markdown(prompt)

    choice_key = f"{key}_choice"
    if choice_key not in st.session_state:
        st.session_state[choice_key] = previous
    current = st.session_state[choice_key]

    left, mid, right = st.columns([1, 6, 1])
    left.caption(SCALE_MIN_LABEL)
    right.caption(SCALE_MAX_LABEL)

    with mid:
        clicked_value = None
        for i, col in enumerate(st.columns(SCALE_POINTS)):
            value = i + 1
            is_selected = current == value
            is_previous = previous is not None and previous == value and not is_selected
            container_key = f"{key}_pill_{value}"

            with col:
                with st.container(key=container_key, border=False):
                    if st.button(
                        str(value),
                        key=f"{key}_btn_{value}",
                        type="primary" if is_selected else "secondary",
                        use_container_width=True,
                    ):
                        clicked_value = value
                if is_previous:
                    st.markdown(
                        f"<style>.st-key-{container_key} button {{"
                        "background-color: rgba(255, 171, 0, 0.32) !important;"
                        "border-color: #ffab00 !important;"
                        "font-weight: 700 !important;"
                        "}}</style>",
                        unsafe_allow_html=True,
                    )

        if clicked_value is not None:
            st.session_state[choice_key] = clicked_value
            st.rerun()

    return current


# -------------------------
# Timing helper
# -------------------------

def ms_since(t0: float) -> int:
    return int((time.time() - t0) * 1000)
