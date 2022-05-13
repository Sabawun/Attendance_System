def decrypt(decryptor, final_encryption_array):
    enc2 = final_encryption_array[5:len(final_encryption_array)]
    dec = ""
    temp_array = ""
    if decryptor[0] == decryptor[int(decryptor[0])]:
        index = int(decryptor[0])
    index = int(decryptor[0])
    for i in enc2:
        m = ord(i)
        # W.I.P part commented for repeated keys.
        #if temp_array == chr(m + 3):
            #dec = dec + chr(m - index - 3)
        #else:
        dec = dec + chr(m - index)
        #temp_array = i

    x = len(dec)
    dec2 = dec[0:x - 1]
    return dec2
