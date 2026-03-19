import os
import sys
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (ApplicationBuilder, CommandHandler, MessageHandler,
                          CallbackQueryHandler, ConversationHandler, filters, ContextTypes)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.review_analyzer import ReviewAnalyzer

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Conversation states
AWAITING_REVIEW, AWAITING_RATING = range(2)

# Global analyzer
analyzer = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome = (
        "Welcome to the Review Authenticity Analyzer!\n\n"
        "I can analyse any product or service review and tell you:\n"
        "- Whether it's likely genuine or fake\n"
        "- Sentiment analysis\n"
        "- Suspicious linguistic patterns\n"
        "- Rating consistency check\n\n"
        "Just paste a review and I'll analyse it.\n"
        "Use /help for more options."
    )
    await update.message.reply_text(welcome)
    return AWAITING_REVIEW


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "HOW TO USE:\n\n"
        "1. Paste any review text\n"
        "2. I'll ask for the star rating (optional)\n"
        "3. Get your full analysis report\n\n"
        "COMMANDS:\n"
        "/start - Start the bot\n"
        "/help - Show this message\n"
        "/analyze - Start a new analysis\n"
        "/quick - Skip rating check, analyse text only\n"
    )
    await update.message.reply_text(help_text)


async def receive_review(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    
    if len(text) < 10:
        await update.message.reply_text("Please paste a longer review (at least 10 characters).")
        return AWAITING_REVIEW
    
    context.user_data['review_text'] = text
    
    keyboard = [
        [InlineKeyboardButton("1 Star", callback_data='1'),
         InlineKeyboardButton("2 Stars", callback_data='2'),
         InlineKeyboardButton("3 Stars", callback_data='3')],
        [InlineKeyboardButton("4 Stars", callback_data='4'),
         InlineKeyboardButton("5 Stars", callback_data='5')],
        [InlineKeyboardButton("Skip Rating", callback_data='skip')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"Got it! ({len(text.split())} words)\n\n"
        "What star rating did this review have?",
        reply_markup=reply_markup
    )
    return AWAITING_RATING


async def receive_rating(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    text = context.user_data.get('review_text', '')
    rating_str = query.data
    
    if rating_str == 'skip':
        rating = None
    else:
        rating = int(rating_str)
    
    await query.edit_message_text("Analysing review... Please wait.")
    
    try:
        report = analyzer.format_full_report(text, rating)
        await query.edit_message_text(report)
    except Exception as e:
        await query.edit_message_text(f"Error during analysis: {str(e)}")
    
    return AWAITING_REVIEW


async def quick_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Paste the review text and I'll analyse it immediately (no rating check).")
    context.user_data['quick_mode'] = True
    return AWAITING_REVIEW


async def handle_quick(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get('quick_mode'):
        text = update.message.text.strip()
        if len(text) < 10:
            await update.message.reply_text("Please paste a longer review.")
            return AWAITING_REVIEW
        
        await update.message.reply_text("Analysing...")
        try:
            report = analyzer.format_full_report(text, None)
            await update.message.reply_text(report)
        except Exception as e:
            await update.message.reply_text(f"Error: {str(e)}")
        
        context.user_data['quick_mode'] = False
        return AWAITING_REVIEW
    
    return await receive_review(update, context)


def main():
    global analyzer
    
    print("Loading model...")
    analyzer = ReviewAnalyzer()
    
    # Try to load saved model, otherwise train
    try:
        analyzer.load_model()
    except FileNotFoundError:
        print("No saved model found. Training...")
        analyzer.train()
        analyzer.save_model()
    
    BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
    
    if BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        print("\n" + "=" * 50)
        print("ERROR: Set your Telegram bot token!")
        print("Either:")
        print("  1. Set environment variable: set TELEGRAM_BOT_TOKEN=your_token")
        print("  2. Replace YOUR_BOT_TOKEN_HERE in the code")
        print("=" * 50)
        return
    
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler('start', start),
            CommandHandler('analyze', start),
            CommandHandler('quick', quick_analyze),
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_quick),
        ],
        states={
            AWAITING_REVIEW: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_quick),
            ],
            AWAITING_RATING: [
                CallbackQueryHandler(receive_rating),
            ],
        },
        fallbacks=[
            CommandHandler('start', start),
            CommandHandler('help', help_command),
        ],
    )
    
    app.add_handler(conv_handler)
    app.add_handler(CommandHandler('help', help_command))
    
    print("\nBot is running! Press Ctrl+C to stop.")
    print("Open Telegram and message @Leonard_review_bot")
    app.run_polling()


if __name__ == "__main__":
    main()