import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Sample data
data = {
    "No": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    "Reviewer": ["K***i", "Abdurrasyid", "Estri", "Winata", "Haidar", "A***d", "s***y", "U***n", "Jessica", "safrudin", "masyuri", "T***o", "H***a", "Asmo", "H***y", "APRIL", "Arief", "A***d", "R***o", "T***a", "D***i", "S***i", "cents", "Rio"],
    "Deskripsi": [
        "Order tgl 25, di kirim tanggal 26 itu pun lama sekali 😓 tapi untung seller nya komunikatif , dan benar” tanggung jawab di kirim , dan barang aman sampai rumah , dan aku dstu udah bener” kesel sekali, karena nunggu nya lama banget😓 sebelum Co di tanya barang bisa kirim gak,katanyah bisa, eh pas di tunggu” gk sampe” juga, tapi terbayarkan sudah dengan barang yg datang sesuai , untuk free case dan anti gores thanks , tapi sayang gabisa di pake 😞, karena bukan untuk iPhone 15pm , semoga barang nya awet, dan tingkatkan lagi untuk pengiriman dan barang nya",
        "Barang bener - bener ori resmi indonesia, pengiriman juga cepet, admin ramah dan responsif nanggepin saya yang bawel, recomended seller",
        "terimakasih ada bonus juga jadi seneng.iphone 15 original resmi iBox . sangat sangat tenang kalo make yg resmi. penjual cepat respon dan pengiriman snagat cepat juga.",
        "Barang sesuai Deskripsi, Packing Aman, Proses Cepat, cuma kurir instant berkali-kali reject (25x lebih reject).",
        "pelayanan sangat cepat, barang sesuai deskripsi, garansi resmi indonesia",
        "Respon admin cepet banget, padahal pesen sore tapi langsung dikirim di hari yang sama, udah saya cek juga barang ori resmi indonesia 100% baru dan segel, dapet warna terbaik, next order lagi disini, terimakasih!",
        "mantap top top. saller selalu respon. pengirimannya sangat cepat. Barangnya mulus Berfungsi dengan baik Packaging rapi",
        "sangat bagus hitam nya. enak untuk maen game juga. iPhone resmi indo aman sinyal nya.",
        "ke3x nya beli disini selalu memuaskan. kali ini beli 15 pro max free tempered glass n casing. thankyou!",
        "Alhamdulillah warna nya pas dan bagus. sedikit kesalahan dari awal warna tapi saler nya sangat cepat respon jadi aman barang yg di inginkan deh.",
        "sangat dibikin puas. pelayanan cepat, barang nya juga sesuai dengan dipesan. pengiriman tepat waktu dan rapih.",
        "Barang bagus pengiriman cepat sesuai diskusi. terima kasih recommended.",
        "Mantap iphone 15 pro max",
        "toko terpercaya, barang masih baru, segel, mulus, dan no tipu2, terimakasih kak CS yg sudah cepat merespon pertanyaan sebelum membeli untuk memuluskan transaksi",
        "barang bagus original",
        "Barangnya baguusss bangeeett thankk u ya",
        "Dapat 15 pro max Ibox termurah dan pengiriman express, thanks",
        "Lgsg dikirim dihari transaksi, packing top 3 lapis, barang aman seaman amannya.. ;)",
        "Produk sesuai iklan Terima kasih",
        "sip mantaap................................",
        "Barang original",
        "Kaya film, Top Gun🤭",
        "-",
        "Mantap sudah langganan"
    ],
    "Rating": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
}

# Load data into a DataFrame
df = pd.DataFrame(data)

# Initialize the tokenizer, stop words, and stemmer
vectorizer = CountVectorizer(tokenizer=word_tokenize, stop_words=stopwords.words('indonesian'))
stemmer = PorterStemmer()

# Tokenization
df['Tokenized'] = df['Deskripsi'].apply(lambda x: word_tokenize(x))

# Filtering: Remove stop words and non-alphabetic tokens
df['Filtered'] = df['Tokenized'].apply(lambda x: [word for word in x if word.isalpha() and word not in stopwords.words('indonesian')])

# Stemming
df['Stemmed'] = df['Filtered'].apply(lambda x: [stemmer.stem(word) for word in x])

# Save to HTML file
df.to_html("reviews.html")

print("Data has been saved to reviews.html")
