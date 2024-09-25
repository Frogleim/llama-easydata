import random


# Function to generate random text with a similar structure
def generate_random_text():
    sample_text = """
    Егейох й1е://Лтр/{random_num}-254Г{random_num2}-ба96-4саЬ-а...

    Управление Федеральной службы судебных приставов по Республике .-• : ,

    Алтай ' \\ — —Г 1 ! ' ; ! : !

    Кош—Агачское районное отделение судебных приставов УФССП России | ! ! ! 1! ! ! ! ! 1 ! ! !

    по Республике Алтай —9 - —Ъ „**3 "! 7" п" " о"’

    ул. Чуйская, д. 8, с. Кош-Агач, Кош- Агачский р-он, Алтай республика, Россия, 649780 " ”

    от 07.03.2023 № 04009/23/43000 Тел. +7(3882)24-96-96

    Для участия в исполнительном производстве воспользуйтесь сервисом Апря://мм.вовиив.ги

    Помер телефона группы телефонного обслуживания указан на официальном сайте ФССП России Ар:/яяр.дом.ги

    Получатель: АО "Яндекс Банк"

    Адрес: ул. Льва Толстого, д. 16, г Москва, 115035

    140008466002 Постановление о розыске счетов и наложении ареста на ДС

    07.03.2023 с. Кош-Агач

    Судебный пристав-исполнитель Кош- Агачское РОСП (Код по ВКСП: 04009), адрес подразделения: 649780,

    Россия, Республика Алтай, Кош- Агачский р-он, , с. Кош-Агач, ул. Чуйская, д. 8, , , Тадинова Татьяна Васильевна,

    рассмотрев материалы сводного исполнительного производства № {random_ip}/21/04009—СД в отношении должника (тип

    должника: физическое лицо): {random_name}, ИНН {random_inn}, д.р. {random_date}, м.р. , , Респ. Алтай,

    Кош-Агачский р-н,, с. Мухор-Тархата,, , „ СНИЛС {random_snils}, адрес должника: 649780, Россия, Респ. Алтай, Кош—

    Агачский р-н,, с. Кош-Агач, ул. Новочуйская, 54, „ на общую сумму {random_amount} р.

    Взыскание выполняется в рамках следующих ИП (количество ИП в сводном - 28):
    """

    # Replace placeholders with random values
    sample_text = sample_text.format(
        random_num=random.randint(10000000000000, 99999999999999),
        random_num2=random.randint(1000, 9999),
        random_ip=random.randint(10000, 99999),
        random_name=random.choice(["Енчинов Элее Германович", "Иванов Иван Иванович"]),
        random_inn=random.randint(100000000000, 999999999999),
        random_date="{:02d}.{:02d}.{}".format(random.randint(1, 28), random.randint(1, 12), random.randint(1950, 2005)),
        random_snils=random.randint(10000000000, 99999999999),
        random_amount=round(random.uniform(1000, 100000), 2)
    )

    return sample_text


# Function to save the generated text to a file
def save_to_txt(filename, text):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)


# Generate random text
random_text = generate_random_text()

# Save the generated text to a txt file
save_to_txt('generated_text.txt', random_text)

print("Text generated and saved to 'generated_text.txt'")
