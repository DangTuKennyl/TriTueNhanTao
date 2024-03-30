import cv2
import face_recognition
import sys

# Tải ảnh lên
anhInput = face_recognition.load_image_file("pic/elon musk.jpg")
anhCheck = face_recognition.load_image_file("pic/EmPhuc.jpg")

# Chuyển ảnh sang định dạng RGB
anhInputRGB = cv2.cvtColor(anhInput, cv2.COLOR_BGR2RGB)
anhCheckRGB = cv2.cvtColor(anhCheck, cv2.COLOR_BGR2RGB)

# Sử dụng thuật toán HOG để nhận diện khuôn mặt
khuonMatInput = face_recognition.face_locations(anhInputRGB, model="hog")
print(khuonMatInput)
khuonMatCheck = face_recognition.face_locations(anhCheckRGB, model="hog")
print(khuonMatCheck)
# Kiểm tra xem có ít nhất một khuôn mặt được tìm thấy trong mỗi ảnh hay không
if len(khuonMatInput) > 0 and len(khuonMatCheck) > 0:
    # Mã hóa hình ảnh khuôn mặt
    encodeInput = face_recognition.face_encodings(anhInputRGB, khuonMatInput)[0]
    encodeCheck = face_recognition.face_encodings(anhCheckRGB, khuonMatCheck)[0]

    # Vẽ hình chữ nhật xung quanh các khuôn mặt được phát hiện
    cv2.rectangle(anhInput, (khuonMatInput[0][3], khuonMatInput[0][0]),
                  (khuonMatInput[0][1], khuonMatInput[0][2]), (255, 0, 255), 2)

    cv2.rectangle(anhCheck, (khuonMatCheck[0][3], khuonMatCheck[0][0]),
                  (khuonMatCheck[0][1], khuonMatCheck[0][2]), (255, 0, 255), 2)

    # So sánh các khuôn mặt
    ketQua = face_recognition.compare_faces([encodeInput], encodeCheck)
    khoangCachKhuonMat = face_recognition.face_distance([encodeInput], encodeCheck)
    print(ketQua,khoangCachKhuonMat)

    # Hiển thị kết quả trên ảnh
    cv2.putText(anhCheck, f"{ketQua} {round(khoangCachKhuonMat[0], 2)}", (khuonMatCheck[0][3], khuonMatCheck[0][0]),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    # Hiển thị ảnh với hình chữ nhật xung quanh khuôn mặt
    cv2.imshow("Input", anhInput)
    cv2.imshow("Kiem_Tra_Khuon_Mat", anhCheck)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Không tìm thấy khuôn mặt trong ít nhất một trong hai ảnh.")
    sys.exit(1)
