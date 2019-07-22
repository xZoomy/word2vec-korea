#delete all comments : \"(.*?)\".*


#good regex : loc_.{16}|qword_.{16}|[+\-\s][a-fA-F0-9]{1,}h
#regex 1 replace by XXXX : loc_.{16}|qword_.{16}
#regex 2 replace by : XXXXh : [+\-\s][a-fA-F0-9]{1,}h

import os,re


 # Read in the file
filedata = None
with open('../input/assembly2.asm', 'r') as file:
  filedata = file.read()

# Replace the target string
filedata.replace('ram', 'abcd')

# Write the file out again
#with open('../input/assembly2.asm', 'w') as file:
#  file.write(filedata)

f = open("../input/assembly2.asm").read()


got = re.findall("Stream: .+\n", f)

got = got[0].strip()

print(got.split(": ")[1])

