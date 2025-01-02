import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import os
import pandas as pd
import plotly.express as px

# Streamlit page configuration
st.set_page_config(
    page_title="Air Pollution Prediction",
    page_icon="💨",
    layout="wide",      
    menu_items={
        "Get Help": "https://github.com/Heysubs",
        "Report a bug": "https://github.com/Heysubs",
        "About": "Website AQI Prediction dengan menggunakan KNN dan SVM yang dibuat oleh Alfin Rifaldi dan Margareta Valencia."
    }
)

# Hide the footer
hide_streamlit_style = """
    <style>
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Dictionary to store model filenames
models = {
    'KNN Model': 'KNN_Model.pkl',
    'SVM Model': 'SVM_Model.pkl'
}

def load_model(filename):
    """Loads a model from a pickle file."""
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file '{filename}' not found.")
        with open(filename, 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except FileNotFoundError as e:
        st.error(f"Error loading the model: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error loading the model: {e}")
        return None

def predict_pollution(model, pm10, pm25, so2, co, o3, no2):
    """Predicts air pollution level using the provided model and input data."""
    try:
        prediction = model.predict([[pm10, pm25, so2, co, o3, no2]])
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def map_pollution_level(prediction):
    pollution_levels = {
        0: ('Baik', 'Tingkat mutu udara yang sangat baik, tidak memberikan efek negatif terhadap manusia, hewan, dan tumbuhan.', 'green'),
        1: ('Sedang', 'Tingkat mutu udara masih dapat diterima pada kesehatan manusia, hewan, dan tumbuhan.', 'blue'),
        2: ('Tidak Sehat', 'Tingkat mutu udara yang bersifat merugikan pada manusia, hewan, dan tumbuhan.', 'yellow'),
        3: ('Sangat Tidak Sehat', 'Tingkat mutu udara yang dapat meningkatkan resiko kesehatan pada sejumlah segmen populasi yang terpapar.', 'red')
    }
    return pollution_levels.get(prediction, ("Tidak Diketahui", "Tidak ada deskripsi yang sesuai.", "gray"))

def load_data(file_path):
    """Reads the CSV file and returns a DataFrame."""
    return pd.read_csv(file_path)

def app():
    # Navigasi Sidebar
    with st.sidebar:
        selected = option_menu('AIR QUALITY INDEX',
                               ['Home',
                                'Data',
                                'Prediction',
                                'Visualization'],
                               default_index=0)

    if selected == 'Home':
        # Judul dan Informasi mengenai Menu EDA
        st.title('Analisis dan Prediksi Indeks Kualitas Udara')
        st.write("""Selamat datang di aplikasi kami yang didedikasikan untuk menganalisis dan memprediksi kualitas udara. 
                 Aplikasi ini memberikan wawasan mendalam tentang kualitas udara di berbagai distrik berdasarkan data historis. 
                 Kami menggunakan teknik Exploratory Data Analysis (EDA) untuk mengidentifikasi pola, tren, dan anomali dalam data kualitas udara. 
                 Dengan prediksi ini, kami berharap dapat membantu meningkatkan kesadaran masyarakat dan mendukung keputusan untuk meningkatkan kualitas udara di masa depan. 
                 Jelajahi berbagai fitur yang tersedia untuk mendapatkan pemahaman yang lebih baik tentang kualitas udara.""")

        st.image('image/polusi.webp', caption='Polusi Udara', use_container_width=True)

        with st.expander("Polusi Udara"):
                st.write("""
            Polusi udara adalah kontaminasi udara oleh zat-zat berbahaya yang menyebabkan dampak negatif 
            terhadap kesehatan manusia dan lingkungan. Zat-zat ini termasuk partikel (PM10 dan PM2.5), 
            gas-gas berbahaya seperti nitrogen dioksida (NO2), sulfur dioksida (SO2), karbon monoksida (CO),
            dan ozon (O3). Di Jakarta, polusi udara menjadi perhatian utama karena tingginya tingkat urbanisasi 
            dan industrialisasi. Sumber polusi udara di kota ini bervariasi, termasuk emisi kendaraan bermotor,
            aktivitas industri, pembakaran bahan bakar fosil, dan bahkan polusi lintas batas dari negara 
            tetangga. Dampak polusi udara pada kesehatan manusia meliputi gangguan pernapasan, penyakit 
            kardiovaskular, dan peningkatan risiko kanker paru-paru. Selain itu, polusi udara juga dapat 
            merusak ekosistem, mengurangi visibilitas, dan menyebabkan kerusakan pada bangunan serta 
            infrastruktur. Melalui aplikasi ini, kami menyediakan data dan analisis terkini mengenai kualitas 
            udara di Jakarta, membantu masyarakat untuk memahami tingkat polusi dan mengambil tindakan pencegahan
            yang diperlukan.
            """)
    
        with st.expander("Indeks Standar Pencemaran Udara (ISPU) "):
                st.write("""
        ISPU merupakan angka tanpa satuan, digunakan untuk menggambarkan kondisi mutu udara ambien di lokasi tertentu dan didasarkan kepada dampak terhadap kesehatan manusia, nilai estetika dan makhluk hidup lainnya. 
                         Khusus untuk daerah rawan terdampak kebakaran hutan dan lahan, informasi ini dapat digunakan sebagai early warning system atau sistem peringatan dini bagi masyarakat sekitar. 
                         Tujuan disusunnya ISPU agar memberikan kemudahan dari keseragaman informasi mutu udara ambien kepada masyarakat di lokasi dan waktu tertentu serta sebagai bahan pertimbangan dalam melakukan upaya-upaya pengendalian pencemaran udara baik bagi pemerintah pusat maupun pemerintah daerah. 
        """)
        
        st.divider()

    # Displaying the table with st.markdown
        st.subheader("Tabel Kategori Indeks Standar Pencemar Udara (ISPU)")
        st.write("""
    Tabel di bawah ini menunjukkan kategori Indeks Standar Pencemar Udara (ISPU) berdasarkan Kementrian Lingkungan Hidup dan Kehutanan :
    """)

        st.write("")
        st.markdown("""
    <table>
        <tr>
            <th>Level</th>
            <th>Value</th>
            <th>Deskripsi</th>
        </tr>
        <tr style='background-color:#7EF490; color:Black'>
            <td>Baik</td>
            <td>0 - 50</td>
            <td>Tingkat mutu udara yang sangat baik, tidak memberikan efek negatif terhadap manusia, hewan, dan tumbuhan</td>
        </tr>
        <tr style='background-color:#87AFF5; color:Black'>
            <td>Sedang</td>
            <td>51 - 100</td>
            <td>Tingkat mutu udara masih dapat diterima pada kesehatan manusia, hewan, dan tumbuhan</td>
        </tr>
        <tr style='background-color:#ECDD57; color:Black'>
            <td>Tidak Sehat</td>
            <td>101 - 200</td>
            <td>Tingkat mutu udara yang bersifat merugikan pada manusia, hewan, dan tumbuhan</td>
        </tr>
        <tr style='background-color:#E93C3C; color:Black'>
            <td rowspan=2>Sangat Tidak</td>
            <td>201 - 300</td>
            <td>Tingkat mutu udara yang dapat meningkatkan resiko kesehatan pada sejumlah segmen populasi yang terpapar</td>
        </tr>
    </table>

    """, unsafe_allow_html=True)
#=========================================================================================    
    st.divider()
    # Dataset
    # Membaca file CSV ke dalam DataFrame
    file_path = "dataset/ispu_jakarta.csv" # Sebelum Handling Outlier
    # file_path = "datasets/df_final.csv" # Setelah Handling Outlier
    def load_data(file_path, index_col=None):
    # Read the CSV file and return a DataFrame
        return pd.read_csv(file_path, index_col=index_col)
    # Now you can call load_data
    df = load_data(file_path)
    if selected == 'Data':
    # Menampilkan data CSV dalam tabel jika ada data yang valid
        if  df is not None:
            st.title('Data Polusi Udara di DKI Jakarta')
            st.write("""
                    Dataset ini mencakup data polusi udara di Jakarta yang **telah diolah** secara komprehensif. Beberapa tahapan yang dilakukan dalam pengolahan dataset ini meliputi:
                    - **Perhitungan AQI (Air Quality Index)**: Indeks Kualitas Udara dihitung berdasarkan konsentrasi berbagai polutan seperti PM2.5, PM10, NO2, SO2, CO, dan O3. AQI memberikan gambaran tentang seberapa bersih atau tercemarnya udara di lokasi tertentu.
                    - **Pembersihan Data**: Tahap ini melibatkan penghapusan data yang hilang atau tidak valid, serta penanganan outlier untuk memastikan kualitas data yang lebih akurat.
                    - **Transformasi Data**: Data mentah diubah menjadi format yang lebih sesuai untuk analisis, termasuk agregasi data berdasarkan waktu atau lokasi.
                    Dengan data yang telah diolah ini, kami dapat melakukan berbagai analisis mendalam dan prediksi mengenai kualitas udara di Jakarta. Data ini sangat penting untuk memahami tren polusi udara, mengidentifikasi faktor-faktor penyebab, dan membantu dalam pengambilan keputusan untuk perbaikan kualitas udara di masa depan.
                """)
            st.dataframe(df, use_container_width=True)
            #url = "https://www.kaggle.com/datasets/bappekim/air-pollution-in-Jakarta"
        #st.caption("Data mentah dapat diperoleh dari KAGGLE [Air Pollution in Jakarta](%s)" % url)
    #else:
        #st.write("Silakan download dataset terlebih dahulu.")

            st.subheader("Deskripsi Variabel Dataset")
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["Tanggal", "PM10", "PM2.5", "SO2", "CO", "O3", "NO2",
                                                  "AQI", "AQI Category", "Stasiun"])

        with tab1:
            st.info("""Tanggal dan waktu saat pengamatan dilakukan.""")
        with tab2:
            st.info("""Partikulat kasar dengan diameter kurang dari 10 mikrometer, 
                berasal dari debu jalan, konstruksi, pembakaran bahan bakar fosil.""")
        with tab3:
            st.info("""Partikulat halus dengan diameter kurang dari 2.5 mikrometer, 
                berasal dari asap kendaraan, pabrik, pembakaran biomassa.""")
        with tab4:    
            st.info("""Sulfur dioksida, berasal dari pembakaran bahan bakar fosil 
                dan aktivitas vulkanik, dapat menyebabkan iritasi saluran pernapasan.""")
        with tab5:
            st.info("""Karbon monoksida, gas tidak berwarna dan tidak berbau yang 
                berasal dari pembakaran tidak sempurna bahan bakar fosil, 
                berbahaya jika terhirup dalam jumlah besar.""")
        with tab6:
            st.info("""Ozon troposferik, terbentuk dari reaksi kimia antara polutan lain 
                di bawah sinar matahari, dapat menyebabkan masalah pernapasan.""")
        with tab7:
            st.info("""Nitrogen dioksida, berasal dari emisi kendaraan bermotor dan 
                pembakaran bahan bakar fosil, dapat menyebabkan iritasi paru-paru.""")
        with tab8:
            st.info("""Indeks yang digunakan untuk melaporkan kualitas udara harian.""")
        with tab9:    
            st.info("""Mengelompokkan kualitas udara ke dalam beberapa tingkat, 
                yang masing-masing memiliki implikasi kesehatan.""")
        with tab10:
            st.info("""Lokasi pengukuran di stasiun.""")
#=========================================================================================
    if selected == 'Prediction' :
        # Page title
        st.header('Prediksi Indeks Kualitas Udara')
        st.subheader('Masukkan nilai polutan berikut:')

        # Input fields for pollutant data
        col1, col2 = st.columns(2)
        with col1:
            pm10 = st.number_input('PM10 (µg/m³)', min_value=0.0, max_value=600.0, format="%.3f")
            pm25 = st.number_input('PM2.5 (µg/m³)', min_value=0.0, max_value=500.0, format="%.3f")
            so2 = st.number_input('SO2 (ppm)', min_value=0.0, max_value=1.0, format="%.3f")
        with col2:
            co = st.number_input('CO (ppm)', min_value=0.0, max_value=2.0, format="%.3f")
            o3 = st.number_input('O3 (ppm)', min_value=0.0, max_value=0.6, format="%.3f")
            no2 = st.number_input('NO2 (ppm)', min_value=0.0,  max_value=50.0, format="%.3f")

        st.write(f"Input data: PM10={pm10}, PM2.5={pm25}, SO2={so2}, CO={co}, O3={o3}, NO2={no2}")

        # Model selection
        selected_model_name = st.selectbox('Pilih model', list(models.keys()))

        # Load the selected model
        model_filename = models[selected_model_name]
        classifier = load_model(model_filename)

        # Define the prediction function
        def predict_pollution(model, pm10, pm25, so2,  co, o3, no2):
            if classifier is not None:
                prediction = classifier.predict([[pm10, pm25, so2,  co, o3, no2]])
                return prediction
            else:
                raise ValueError("The classifier model is not loaded.")

        # Prediction button
        if st.button('Predict'):
            prediction = predict_pollution(classifier, pm10, pm25, so2, co, o3, no2)
            if prediction is not None:
                pollution_level, description, color = map_pollution_level(prediction[0])
                st.markdown(f"<h4 style='color:{color}'>Kategori kualitas udara yang diprediksi adalah: {pollution_level}</h4>", unsafe_allow_html=True)
                st.info(description)

    if selected == 'Visualization':
        # Load the data
        df = load_data('dataset/ispu_jakarta.csv')
        
        # Make sure the 'tanggal' column is in datetime format
        df['tanggal'] = pd.to_datetime(df['tanggal'])
        
        # Set up a sidebar with filters (Optional)
        st.sidebar.header("Filters")
        stasiun_filter = st.sidebar.selectbox("Pilih stasiun", df['stasiun'].unique())
        pollutant_filter = st.sidebar.selectbox("Pilih Polutan", ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2'])
        
        # Filter data based on stasiun selection
        df_filtered = df[df['stasiun'] == stasiun_filter]
        
        # Create a time-series plot for AQI or a selected pollutant over time
        st.header(f"Air Quality Trend for {stasiun_filter}")
        
        # Plot AQI or selected pollutant
        if pollutant_filter == "AQI":
            fig = px.line(df_filtered, x="tanggal", y="AQI", title=f"Time Series of AQI in {stasiun_filter}")
        else:
            fig = px.line(df_filtered, x="tanggal", y=pollutant_filter, title=f"Time Series of {pollutant_filter} in {stasiun_filter}")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show distribution of pollutant levels
        st.header(f"Distribution of {pollutant_filter} Levels in {stasiun_filter}")
        fig_hist = px.histogram(df_filtered, x=pollutant_filter, nbins=50, title=f"Distribution of {pollutant_filter} Levels")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # AQI category distribution
        st.header(f"AQI Category Distribution in {stasiun_filter}")
        category_counts = df_filtered['categori'].value_counts().reset_index()
        category_counts.columns = ['categori', 'Count']
        
        fig_bar = px.bar(category_counts, x='categori', y='Count', title=f"AQI Category Distribution")
        st.plotly_chart(fig_bar, use_container_width=True)


if __name__ == "__main__":
    app()
