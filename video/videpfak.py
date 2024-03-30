import streamlit as st
import os
import shutil
from finalpredicted import predict_deepfake

def main():
    st.title("DeepFake Detection App")
    st.write("Upload a video file and select the method for deepfake detection.")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

    if uploaded_file is not None:
        method = st.selectbox("Select method", ["plain_frames"])

        if st.button("Detect DeepFake"):
            with st.spinner("Detecting..."):
                input_videofile_path = "uploaded_video.mp4"
                with open(input_videofile_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                fake_prob, real_prob, pred = predict_deepfake(input_videofile_path, method)

            if pred is None:
                st.error("Failed to detect DeepFakes")
            else:
                label = "real" if pred == 0 else "deepfaked"
                probability = real_prob if pred == 0 else fake_prob
                probability = round(probability * 100, 4)

                if pred == 0:
                    st.success(f"The video is {label},with a probability of:{probability}%")
                    shutil.rmtree("./output")  
                else:
                    st.error(f"The video is {label},with a probability of :{probability}%")
                    shutil.rmtree("./output")  

if __name__ == "__main__":
    main()
