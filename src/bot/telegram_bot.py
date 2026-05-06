import os
import sys
import logging
import pickle
import numpy as np
import re
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (ApplicationBuilder, CommandHandler, MessageHandler,
                          CallbackQueryHandler, ConversationHandler, filters, ContextTypes)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.models.linguistic_detector import LinguisticPatternDetector
from src.models.type_detector import FakeReviewTypeDetector

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

AWAITING_REVIEW, AWAITING_RATING = range(2)

BOT_TOKEN = "8651836361:AAELCAElIx5PaC-SRdSjNp3pi0GaFF3BU2Y"

FEATURE_COLS = [
    'char_count', 'word_count', 'sentence_count', 'avg_word_length',
    'avg_sentence_length', 'unique_words', 'lexical_diversity',
    'first_person_singular', 'first_person_plural', 'first_person_total',
    'first_person_ratio', 'exclamation_count', 'question_count',
    'capital_ratio', 'superlative_count', 'superlative_ratio',
    'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral',
    'number_count', 'number_ratio', 'stopword_ratio',
    'max_word_freq', 'repeated_words_ratio'
]

STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
}

# Global components
classifier = None
tfidf = None
label_encoder = None
sentiment_analyzer = None
linguistic_detector = None


def extract_features(text):
    words = re.findall(r'[a-zA-Z]+', text.lower())
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words_no_stop = [w for w in words if w not in STOP_WORDS]
    word_count = len(words)
    sentence_count = len(sentences)
    
    if word_count == 0:
        return {col: 0 for col in FEATURE_COLS}
    
    superlatives = ['best', 'worst', 'amazing', 'terrible', 'perfect', 'horrible',
                    'excellent', 'awful', 'fantastic', 'disgusting', 'wonderful',
                    'dreadful', 'outstanding', 'pathetic', 'incredible', 'unbelievable',
                    'fabulous', 'atrocious', 'superb', 'abysmal', 'magnificent',
                    'absolutely', 'totally', 'completely', 'utterly', 'extremely',
                    'never', 'always', 'every', 'nothing', 'everything']
    
    sentiment = sentiment_analyzer.analyze(text)
    fp_singular = sum(1 for w in words if w in ['i', 'me', 'my', 'mine', 'myself'])
    fp_plural = sum(1 for w in words if w in ['we', 'us', 'our', 'ours', 'ourselves'])
    sup_count = sum(1 for w in words if w in superlatives)
    
    word_freq = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1
    
    return {
        'char_count': len(text),
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': np.mean([len(w) for w in words]),
        'avg_sentence_length': word_count / sentence_count if sentence_count > 0 else 0,
        'unique_words': len(set(words)),
        'lexical_diversity': len(set(words)) / word_count,
        'first_person_singular': fp_singular,
        'first_person_plural': fp_plural,
        'first_person_total': fp_singular + fp_plural,
        'first_person_ratio': (fp_singular + fp_plural) / word_count,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'capital_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
        'superlative_count': sup_count,
        'superlative_ratio': sup_count / word_count,
        'vader_compound': sentiment['compound'],
        'vader_positive': sentiment['positive'],
        'vader_negative': sentiment['negative'],
        'vader_neutral': sentiment['neutral'],
        'number_count': len(re.findall(r'\d+', text)),
        'number_ratio': len(re.findall(r'\d+', text)) / word_count,
        'stopword_ratio': (word_count - len(words_no_stop)) / word_count,
        'max_word_freq': max(word_freq.values()),
        'repeated_words_ratio': sum(1 for v in word_freq.values() if v > 1) / len(word_freq),
    }


def analyze_review(text, rating=None):
    from scipy.sparse import hstack, csr_matrix
    
    features = extract_features(text)
    X_eng = np.array([[features[col] for col in FEATURE_COLS]])
    X_tfidf = tfidf.transform([text])
    X_combined = hstack([X_tfidf, csr_matrix(X_eng)])
    
    proba = classifier.predict_proba(X_combined)[0]
    classes = list(label_encoder.classes_)
    fake_prob = proba[classes.index('fake')]
    genuine_prob = proba[classes.index('genuine')]
    authenticity_score = round(genuine_prob * 100, 1)
    
    if authenticity_score >= 70:
        verdict = "LIKELY GENUINE ✅"
    elif authenticity_score >= 40:
        verdict = "UNCERTAIN ⚠️"
    else:
        verdict = "LIKELY FAKE ❌"
    
    sentiment = sentiment_analyzer.analyze(text)
    patterns = linguistic_detector.detect(text)
    
    lines = []
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("📊 REVIEW ANALYSIS REPORT")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("")
    lines.append(f"🔍 Verdict: {verdict}")
    lines.append(f"📈 Authenticity Score: {authenticity_score}%")
    lines.append(f"   Genuine: {round(genuine_prob*100,1)}% | Fake: {round(fake_prob*100,1)}%")
    
    lines.append("")
    lines.append(f"💭 Sentiment: {sentiment['label'].upper()} ({sentiment['intensity']})")
    lines.append(f"   Score: {sentiment['compound']} (Pos: {sentiment['positive']} | Neg: {sentiment['negative']})")
    
    if rating is not None:
        consistency = sentiment_analyzer.check_rating_consistency(text, rating)
        lines.append("")
        if consistency['consistent']:
            lines.append(f"⭐ Rating Check: Consistent with {rating}-star rating ✅")
        else:
            lines.append(f"⭐ Rating Check: {consistency['message']}")
    
    if patterns['flags']:
        lines.append("")
        lines.append(f"🚩 Suspicious Patterns ({patterns['flag_count']} found):")
        for flag in patterns['flags']:
            icon = "🔴" if flag['severity'] == 'HIGH' else "🟡"
            lines.append(f"   {icon} {flag['pattern']}")
    else:
        lines.append("")
        lines.append("🚩 No suspicious patterns detected ✅")
    
    if authenticity_score < 60:
        type_result = type_detector.detect_type(text, sentiment=sentiment)
        if type_result['type'] != 'none_detected':
            lines.append("")
            lines.append(f"🏷️ Likely Type: {type_result['type_label']}")
            lines.append(f"   Confidence: {type_result['confidence']}%")
            for r in type_result['reasoning']:
                lines.append(f"   • {r}")

    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    return "\n".join(lines)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome = (
        "👋 Welcome to the Review Authenticity Analyzer!\n\n"
        "I use NLP and machine learning to detect fake reviews.\n\n"
        "📝 Just paste any product review and I'll analyse it.\n\n"
        "Commands:\n"
        "/start - Restart\n"
        "/help - How to use\n"
        "/about - About this bot"
    )
    await update.message.reply_text(welcome)
    return AWAITING_REVIEW


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "📖 HOW TO USE\n\n"
        "1️⃣ Paste any review text\n"
        "2️⃣ Select the star rating (or skip)\n"
        "3️⃣ Get your analysis report\n\n"
        "The report includes:\n"
        "• Authenticity score (0-100%)\n"
        "• Fake/genuine classification\n"
        "• Sentiment analysis\n"
        "• Rating consistency check\n"
        "• Suspicious pattern detection\n\n"
        "Tip: Include the star rating for a more complete analysis!"
    )
    await update.message.reply_text(help_text)
    return AWAITING_REVIEW


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    about = (
        "🤖 ABOUT THIS BOT\n\n"
        "NLP-Based Detection and Authenticity Scoring of "
        "Fake Reviews in E-Commerce\n\n"
        "🔬 Model: Random Forest + TF-IDF\n"
        "📊 Accuracy: 88.6% F1-Score\n"
        "📚 Trained on: Ott et al. (2011) + Amazon Reviews\n"
        "🔧 Features: 25 linguistic + 5000 TF-IDF\n\n"
        "Developed as a Masters dissertation project."
    )
    await update.message.reply_text(about)
    return AWAITING_REVIEW


async def receive_review(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    
    
    if len(text) < 15:
        await update.message.reply_text("⚠️ Please paste a longer review (at least 15 characters).")
        return AWAITING_REVIEW
    
    if len(text) > 5000:
        text = text[:5000]
        await update.message.reply_text("📝 Review truncated to 5000 characters.")
    
    context.user_data['review_text'] = text
    word_count = len(text.split())
    
    keyboard = [
        [InlineKeyboardButton("⭐ 1", callback_data='1'),
         InlineKeyboardButton("⭐ 2", callback_data='2'),
         InlineKeyboardButton("⭐ 3", callback_data='3'),
         InlineKeyboardButton("⭐ 4", callback_data='4'),
         InlineKeyboardButton("⭐ 5", callback_data='5')],
        [InlineKeyboardButton("⏩ Skip Rating", callback_data='skip')],
    ]
    
    await update.message.reply_text(
        f"📝 Received review ({word_count} words)\n\n"
        "What star rating does this review have?",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return AWAITING_RATING


async def receive_rating(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    text = context.user_data.get('review_text', '')
    rating_str = query.data
    rating = None if rating_str == 'skip' else int(rating_str)
    
    await query.edit_message_text("🔄 Analysing review... Please wait.")
    
    try:
        report = analyze_review(text, rating)
        await query.edit_message_text(report)
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        await query.edit_message_text(f"❌ Error during analysis. Please try again.\n\nDetail: {str(e)[:200]}")
    
    return AWAITING_REVIEW


def load_models():
    global classifier, tfidf, label_encoder, sentiment_analyzer, linguistic_detector
    global type_detector
    type_detector = FakeReviewTypeDetector()
    print("  Type detector loaded")
    
    print("Loading models...")
    
    model_dir = "models"
    
    # Try combined model first, fall back to original
    try:
        with open(os.path.join(model_dir, "combined_classifier.pkl"), "rb") as f:
            classifier = pickle.load(f)
        with open(os.path.join(model_dir, "combined_tfidf.pkl"), "rb") as f:
            tfidf = pickle.load(f)
        with open(os.path.join(model_dir, "combined_label_encoder.pkl"), "rb") as f:
            label_encoder = pickle.load(f)
        print("  Combined model loaded (Ott + Amazon)")
    except FileNotFoundError:
        with open(os.path.join(model_dir, "classifier.pkl"), "rb") as f:
            classifier = pickle.load(f)
        with open(os.path.join(model_dir, "tfidf.pkl"), "rb") as f:
            tfidf = pickle.load(f)
        with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
            label_encoder = pickle.load(f)
        print("  Original model loaded (Ott only)")
    
    sentiment_analyzer = SentimentAnalyzer()
    linguistic_detector = LinguisticPatternDetector()
    print("  Sentiment analyzer loaded")
    print("  Linguistic detector loaded")
    print("All models ready!")


def main():
    load_models()
    
    print("\nStarting bot...")
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler('start', start),
            MessageHandler(filters.TEXT & ~filters.COMMAND, receive_review),
        ],
        states={
            AWAITING_REVIEW: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_review),
            ],
            AWAITING_RATING: [
                CallbackQueryHandler(receive_rating),
            ],
        },
        fallbacks=[
            CommandHandler('start', start),
            CommandHandler('help', help_command),
            CommandHandler('about', about_command),
        ],
    )
    
    app.add_handler(conv_handler)
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('about', about_command))
    
    print("\n✅ Bot is running!")
    print("Open Telegram and message @NWATU_review_bot")
    print("Press Ctrl+C to stop.\n")
    
    app.run_polling()


if __name__ == "__main__":
    main()