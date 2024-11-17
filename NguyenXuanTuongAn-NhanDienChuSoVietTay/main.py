from tkinter import *
import cv2
import numpy as np
from PIL import ImageGrab, Image
from tensorflow.keras.models import load_model
from tkinter import filedialog  # Thêm thư viện cho phép mở hộp thoại chọn file

model = load_model('mnist.h5')

root = Tk()
root.resizable(0, 0)
root.title("HDR")

lastx, lasty = None, None

cv = Canvas(root, width=640, height=480, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)


def clear_widget():
    global cv
    cv.delete('all')


def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y


def activate_event(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y


cv.bind('<Button-1>', activate_event)


def Recognize_Digit():
    widget = cv

    x = widget.winfo_rootx()
    y = widget.winfo_rooty()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    # Capture the image directly without saving
    captured_image = ImageGrab.grab().crop((x, y, x1, y1))

    # Convert the image to OpenCV format
    image = cv2.cvtColor(np.array(captured_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # make a rectangle box around each curve
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = th[y:y + h, x:x + w]

        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (18, 18))

        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

        digit = padded_digit.reshape(1, 28, 28, 1)
        digit = digit / 255.0

        pred = model.predict([digit])[0]
        final_pred = np.argmax(pred)

        data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)

    cv2.imshow('image', image)
    cv2.waitKey(0)


def upload_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")))
    
    if file_path:
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Làm mịn ảnh để giảm nhiễu
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Ngưỡng để chuyển sang ảnh nhị phân
        ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Tìm các đường viền
        contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Bỏ qua các vùng có kích thước nhỏ hoặc hình dạng không hợp lý
            if w > 20 and h > 20:  # Điều chỉnh kích thước theo nhu cầu
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

                digit = th[y:y + h, x:x + w]

                # Định cỡ chữ số và căn giữa
                resized_digit = cv2.resize(digit, (18, 18))
                padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

                digit = padded_digit.reshape(1, 28, 28, 1)
                digit = digit / 255.0

                pred = model.predict([digit])[0]
                final_pred = np.argmax(pred)

                data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'

                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                color = (255, 0, 0)
                thickness = 1
                cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)

        cv2.imshow('Recognized Image', image)
        cv2.waitKey(0)

btn_save = Button(text='Recognize Digit', command=Recognize_Digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)

button_clear = Button(text='Clear Widget', command=clear_widget)
button_clear.grid(row=2, column=1, pady=1, padx=1)

button_upload = Button(text='Upload Image', command=upload_image)  # Nút tải ảnh lên
button_upload.grid(row=3, column=0, pady=1, padx=1)

root.mainloop()
