import re
import sys

regex1 = r"loc_.{16}|qword_.{16}|byte_.{16}|unk_.{16}|locret_.{16}|dword_.{16}|off_.{16}|asc_.{16}|(?i)[a-f\d]{16}"
regex2 = r"[A-F\d]{2,17}h"
# regexBackup = r"[\s+\-][a-fA-F0-9]{1,}h"

file1 = "./../input/assembly.asm"
file2 = "./../input/test.asm"

replace1 = "XXXX"
replace2 = "XXXXh"

f=open("./../input/test.asm","w+")
with open (file1, "r") as myfile:
     s=myfile.read()
ret1 = re.sub(regex1,replace1, s)   # <<< This is where the magic happens
ret2 = re.sub(regex2,replace2, ret1)
f.write(ret2)
f.close()
