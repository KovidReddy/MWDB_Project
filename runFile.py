import imageProcess
import numpy as np
ip = imageProcess.imageProcess()
arg = input("Which task would you like to run (1/2/3)? ")
if arg == '1':
    for x in ['h', 's', 'l', 'm']:
        print(x)
        ip.dbProcess(model=x, process='s')

elif arg == '2':
    inp1 = input("Would you like to fetch SIFT features or Moment features (s/m)?")
    inp = input("Provide the image ID which you would like to display:")

    rec = ip.dbProcess(model=inp1, process='f')
    rec_arr = np.array(rec[0][1])
    print('Features:', np.array(rec_arr))
    print('Size', rec_arr.shape)
    ip.writeFile(rec[0][1], 'Output\\Task2\\task2_{x}_{y}.txt'.format(x=inp1,
                                                                      y=inp))
    print('File saved to Output folder')

elif arg == '3':
    inp1 = input("Would you like to Compare SIFT features or Moment features (s/m)?")
    inp = input("Provide the image ID which you would like to display:")
    inp2 = input("Provide the number of similar images you would like to return?")

    recs = ip.dbProcess(password='password', model=inp1, process='f')
    sim = ip.SimCalc(inp, recs, imgmodel=inp1, k=int(inp2))
    sim_list = [im[0] for im in sim]
    sim_list.insert(0, inp)
    print('\n')
    print('Top {x} Similar Images'.format(x=inp2))
    
    for x, y in sim:
        print('Image: {x} , Similarity: {y} \n'.format(x=x, y=y))
    ip.writeFile(sim, 'Output\\Task3\\task3_{x}_{y}.txt'.format(x=inp1,
                                                                y=inp))
    ip.dispImages(sim_list, 'Output\\Task3\\{x}_{y}'.format(x=inp, y=inp1))
