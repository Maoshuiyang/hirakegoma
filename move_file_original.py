import shutil
import os
import datetime

#dt_now = datetime.datetime.now()

os.makedirs('./dog-cat/3karasu/wav/angry', exist_ok=True)
os.makedirs('./dog-cat/3karasu/wav/normal', exist_ok=True)
os.makedirs('./dog-cat/3karasu/wav/others', exist_ok=True)
s=0
    
for i in range(23):
    if i%2==0:
        path = 'angry'
    elif i%3==0:
        path = 'normal'
    else:
        path = 'others'
    
    dt_now = datetime.datetime.now()
    with open('./dog-cat/3karasu/wav/file.txt', 'a') as f: #w
        f.write(str(dt_now)+'_'+str(s)+': '+path+'\n')
    #f = open('./dog-cat/3karasu/wav/file.txt','a')
    #f.write(path + '\n')
    #f.close()    
    
    print(os.listdir('./dog-cat/3karasu/wav'))
    # ['file.txt', 'dir']

    print(os.listdir('./dog-cat/3karasu/wav/angry'))
    # []
    
    new_path = shutil.move('./dog-cat/3karasu/wav/' + str(s)+'.wav', './dog-cat/3karasu/wav/'+ path)
    new_path = shutil.move('./dog-cat/3karasu/wav/figure' + str(s)+ '.jpg', './dog-cat/3karasu/wav/' + path)

    print(new_path)
    # temp/dir2/file.txt

    print(os.listdir('./dog-cat/3karasu/wav'))
    # ['dir']

    print(os.listdir('./dog-cat/3karasu/wav/angry'))
    # ['file.txt']
    s += 1