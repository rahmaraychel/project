def emoji_replacer(message):
    emoji_dict = {
        "happy":"😂",
        "sad":"😥",
        "tired":"😒",
        "angry":"🤨"
    }
    words = message.split()
    result = []
    for word in words:
        result.append(emoji_dict.get(word.lower(),word))
    return " ".join(result)
user_message=input("enter your messge:")
print("convert message:",emoji_replacer(user_message))