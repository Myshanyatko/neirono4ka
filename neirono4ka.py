from fastai.vision import *
import telebot

bot = telebot.TeleBot('1212187995:AAGpx72h7PFM_J1ByutPZF5bUGKvpIfNWMk')
@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Салам пополам, это бот, который определит, кто ты из Доты. Нейросеть не ошибается, и не смей с ней спорить. Скидывай фоточку и ожидай в общем. (Фотку передадим в госдеп США потом и кредит оформим да)')

@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):

    file_info = bot.get_file(message.photo[len(message.photo)-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open("image.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)

    path = r'C:\Users\пк\new neirono4ka\2 heroes'
    learn = load_learner(path)
    img = open_image(r'C:\Users\пк\PycharmProjects\neirono4ka\image.jpg')
    pred_class, pred_idx, outputs = learn.predict(img)
    card = r"""{recipient}.jpg"""
    adr = card.format(recipient=pred_class)
    bot.send_message(message.chat.id, 'Не верим своим глазам, вы...')
    f = open(adr, 'rb')
    bot.send_photo(message.chat.id, f)

@bot.message_handler(content_types=['text', 'sticker', 'audio'])
def handle_docs_text(message):
    bot.send_message(message.chat.id, 'ебать это по твоему фотка?')
bot.polling(none_stop=True)
