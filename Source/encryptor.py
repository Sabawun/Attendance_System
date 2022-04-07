from Source.decryptor import decrypt


def encrypt(x):
    encrypted_array = ""
    temp_array = "first"
    for i in x:
        m = ord(i)
        # W.I.P part commented for repeating keys
        #if temp_array == i:
            #encrypted_array = encrypted_array + chr(m + 8)
        #else:
        encrypted_array = encrypted_array + chr(m + 5)
        temp_array = i

    final_encryption_array = ""
    index = 0
    # part 2
    for i in encrypted_array:
        m = ord(i)
        final_encryption_array = final_encryption_array + chr(m + 3)
        index = index + 1
        if index == 5:  # index for decrption and encrption
            break

    final_encryption_array = final_encryption_array + encrypted_array + str(index)
    return final_encryption_array


def decrpyt_key(encrypted_password):
    x = len(encrypted_password)
    index = int(encrypted_password[x - 1])
    final = ""
    key = "tEnbLbjBqgzXhJKPQaklWqMKmGCRvP"
    for i in key:
        m = ord(i)
        final = final + chr(m + index)

    s1 = final[0:index - 1]
    s2 = final[index:len(final)]

    decrpytor = str(index) + s1 + str(index) + s2
    return decrpytor


Pass = "HelloTest__qqq#$"
print("Entered Password is : " + Pass)
encrypted_password = encrypt(Pass)
print("Encrypted Password : " + encrypted_password)
decryption_key = decrpyt_key(encrypted_password)
print("Decryption key : " + decryption_key)
password = decrypt(decryption_key, encrypted_password)
print("Decrypted Password : " + password)
