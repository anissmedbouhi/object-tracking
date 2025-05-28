from time import process_time, perf_counter
import cv2 as cv
import glob
import numpy as np
from utils import *
import os
import multiprocessing
from sklearn.cluster import DBSCAN

processors_num = multiprocessing.cpu_count()


class state():
    def __init__(self,x,y,x_dot,y_dot,h_x,h_y,a_dot):
        self.x=x
        self.y=y
        self.x_dot=x_dot
        self.y_dot=y_dot
        self.h_x=h_x
        self.h_y=h_y
        self.a_dot=a_dot
    def draw_dot(self,img,path):    
        overlay = img.copy()
        cv.circle(overlay, center=(self.x, self.y), radius=1, color=(0, 0, 255), thickness=4)
        transparency = 0.7
        cv.addWeighted(overlay, 1 - transparency, img, transparency, 0, img)
        self.img=img
        cv.imwrite(path, self.img)
    def draw_rectangle(self,img,path, color=(0,0,255)):
        cv.rectangle(img, (self.x - self.h_x,self.y - self.h_y), ( self.x + self.h_x,self.y + self.h_y),color,thickness=1)
        self.img=img
        cv.imwrite(path, self.img)
    def output(self):
        print('x: %s  y: %s  h_x: %s  h_y: %s  x_dot: %s  y_dot: %s  a_dot: %s'%(self.x,self.y,self.h_x,self.h_y,self.x_dot,self.y_dot,self.a_dot))
        ##results_text.write('\nx: %s  y: %s  h_x: %s  h_y: %s  x_dot: %s  y_dot: %s  a_dot: %s'%(self.x,self.y,self.h_x,self.h_y,self.x_dot,self.y_dot,self.a_dot))


class hist():
    def __init__(self, num, max_range=360):
        self.num=num
        self.max_range=max_range
        self.divide=[max_range/num*i for i in range(num)]
        self.height=np.array([0. for i in range(num)])
        self.spacing = max_range/num

    def get_hist_id(self,x):
        return (int(x // self.spacing))

    def update(self,i):
        self.height[i]+=1


class ParticleFilter():
    def __init__(self,particles_num=56,img_path=r'./test',out_path=r'./output', H_num = 8, S_num = 4, V_num = 4, resampling_condition = 4, min_obs_in_cluster = 2, supdate_threshold = 0.97):
        
        
        self.img_Neff_B = []
        
        self.B_saved = [] # Battacharya distances between the estimated state and the target model
        self.B_possible = [] # Battacharya coefficient for the possible states, one state and B coefficient for each cluster
        
        self.resampling_condition = resampling_condition # See in Resampling(), it resamples under the condition (Neff < particles_num/resampling_condition)
        self.min_obs_in_clusters = min_obs_in_clusters # See in Clustering_estimate(), it is the minimum number of observations in one cluster when doing the DBSCAN clustering algo called where this number is called "min_samples"
        
        # Threshold, if B>T then we suppose the object not lost nor occulted
        self.T = update_threshold
        
        self.H_num = H_num
        self.S_num = S_num
        self.V_num = V_num
        
        self.clusters_num = 1
        self.right_cluster = 0
        
        self.particles_num=particles_num
        self.out_path=out_path
        results_text = open(self.out_path+'/results_text.txt', 'a')
        self.DELTA_T=0.05 #originally 0.05
        self.VELOCITY_DISTURB=4.
        self.SCALE_DISTURB=0.0 #originally=0.0 If equal to 0.1, the car is lost at a moment and another car is tracked
        self.SCALE_CHANGE_D=0.001 #originally=0.001 #essais avec 0.1
        self.img_index=0
        #self.imgs=glob.glob(os.path.join(img_path,'*.jpg'))
        #self.imgs=[os.path.join(img_path,'%04d.jpg'%(i+1)) for i in range(40, 551, 10)] #images 40 to 551, 10 by 10
        self.imgs=[os.path.join(img_path,'%04d.jpg'%(i+1)) for i in range(40, 551)]
        print(self.imgs[0])
        results_text.write(str(self.imgs[0]))
        print('processing image: %04d.jpg' % (self.img_index + 1))
        results_text.write('\nprocessing image: %04d.jpg' % (self.img_index + 1))

        img_first = cv.imread(self.imgs[0])
        initial_state=state(x=425,y=220,x_dot=0.,y_dot=0.,h_x=80,h_y=35,a_dot=0.) #à changer par une initialisation avec une loi uniforme pour voir combien il faut d'itérations pour avoir quelque chose d'acceptable
        initial_state.draw_dot(img_first,self.out_path+'/0001.jpg')
        initial_state.draw_rectangle(img_first, self.out_path+'/0001.jpg')
        self.state=initial_state # estimated state
        self.states_possible = [initial_state] # possible states, one state by cluster
        self.particles=[]
        random_nums=np.random.normal(0,0.4,(particles_num,7)) 
        self.weights = [1. / particles_num] * particles_num  #uniform weight on the particles
        self.weights_before_normalization = self.weights
        for i in range(particles_num): # à paralléliser ! Changer self.particles en dictionnaire ?
            x0 = int(initial_state.x + random_nums.item(i, 0) * initial_state.h_x)
            y0 = int(initial_state.y + random_nums.item(i, 1) * initial_state.h_y)
            x_dot0 = initial_state.x_dot + random_nums.item(i, 2) * self.VELOCITY_DISTURB
            y_dot0 = initial_state.y_dot + random_nums.item(i, 3) * self.VELOCITY_DISTURB
            h_x0 = int(initial_state.h_x + random_nums.item(i, 4) * self.SCALE_DISTURB)
            h_y0 = int(initial_state.h_y + random_nums.item(i, 5) * self.SCALE_DISTURB)
            a_dot0 = initial_state.a_dot + random_nums.item(i, 6) * self.SCALE_CHANGE_D
            particle = state(x0, y0, x_dot0, y_dot0, h_x0, h_y0, a_dot0)
            particle.draw_dot(img_first,self.out_path+'/0001.jpg')
            self.particles.append(particle)
        
        #OpenCV uses H: 0-179, S: 0-255, V: 0-255
        self.q = [hist(num=H_num, max_range=180),hist(num=S_num,max_range=255),hist(num=V_num,max_range=255)] # q is the TARGET # from the paper: "the histograms are typically calculated in the RGB space using 8×8×8 bins. To make the algorithm less sensitive to lighting conditions, the HSV color space could be used instead with less sensitivity to V (e.g. 8×8×4 bins).        
        img_first = cv.imread(self.imgs[0])
        img_first = cv.cvtColor(img_first, cv.COLOR_BGR2HSV) # convert BGR to HSV
        self.imgHSV = img_first # in order to access it easily for the parallelization in observe_single
        for hist_c in self.q:
            for u in range(hist_c.num): #hist_c.num va jusqu'à 2, puis 2, puis 10
                a = np.sqrt(initial_state.h_x**2+initial_state.h_y**2)
                weight = []
                x_bin = []
                for i in range(initial_state.x - initial_state.h_x, initial_state.x + initial_state.h_x):
                    for j in range(initial_state.y - initial_state.h_y, initial_state.y + initial_state.h_y):
                        x_val = img_first[j][i][self.q.index(hist_c)]
                        a1 = i - initial_state.x
                        a2 = j - initial_state.y
                        temp = k(np.sqrt(a1*a1 + a2*a2) / a)                        
                        weight.append(temp)
                        x_bin.append(k_delta(hist_c.get_hist_id(float(x_val)) - u))
                hist_c.height[u] = np.sum(np.array(weight) * np.array(x_bin))
        concatenated_hist_q = np.concatenate((self.q[0].height,self.q[1].height,self.q[2].height)) # let double parenthesis
        f_normalization = np.sum(concatenated_hist_q)  
        self.concatenated_hist_q = concatenated_hist_q/f_normalization
        print('concatenated_hist_q: ', self.concatenated_hist_q)

    def select(self): # This is a multinomial resampling
    
        results_text = open(self.out_path+'/results_text.txt', 'a')
    
        if self.img_index<len(self.imgs)-1:
            self.img_index+=1
        self.img = cv.imread(self.imgs[self.img_index])
        print('processing image: %04d.jpg' % (self.img_index+1))
        results_text.write('\nprocessing image: %04d.jpg' % (self.img_index+1))
        index=get_random_index(self.weights)
        new_particles=[]
        for i in index:
            new_particles.append(state(self.particles[i].x,self.particles[i].y,self.particles[i].x_dot,self.particles[i].y_dot,self.particles[i].h_x,self.particles[i].h_y,self.particles[i].a_dot))
        self.particles=new_particles


    def resampling(self): # This is a Systematic Resampling or Stochastic Universal Resampling
        
        results_text = open(self.out_path+'/results_text.txt', 'a')
        
        if self.img_index<len(self.imgs)-1:
            self.img_index+=1
        self.img = cv.imread(self.imgs[self.img_index])
        print('processing image: %04d.jpg' % (self.img_index+1))
        results_text.write('\nprocessing image: %04d.jpg' % (self.img_index+1))
        
        denom=0
        for n in range(self.particles_num):
            denom += (self.weights[n])**2
        Neff = 1/denom
        print('Neff: ', Neff) # Neff measures approximately the number of particles which meaningfully contribute to the probability distribution
        self.img_Neff_B.append(self.img_index)
        self.img_Neff_B.append(Neff)
        
        if Neff < self.particles_num/self.resampling_condition: # I don't resample at each step in order to let some hypothesis, for example I resample if Neff < N/4
            print('resampling...')
            index=[]        
            r=np.random.rand()/self.particles_num # (random number between 0 and 1/self.particles_num)
            c = self.weights[0]
            i = 0
            for n in range(self.particles_num):
                U = r + n/self.particles_num
                while U > c:
                    i+=1
                    c+= self.weights[i]
                index.append(i)
    
            new_particles=[]
            for i in index:
                new_particles.append(state(self.particles[i].x,self.particles[i].y,self.particles[i].x_dot,self.particles[i].y_dot,self.particles[i].h_x,self.particles[i].h_y,self.particles[i].a_dot))
            self.particles=new_particles



    def propagate(self):
        
        for particle in self.particles:
            random_nums = np.random.normal(0, 0.4, 7)
            random_nums[6]=np.random.normal(0, 0.8)

            if (  len(self.B_saved)>0 and self.B_saved[-1]<0.9  ):
                random_nums[0] = np.random.normal(0, 2)
                random_nums[1] = np.random.normal(0, 2)
                random_nums[6]=np.random.normal(0, 0.4)
            '''
            elif (  len(self.B_saved)>0 and self.B_saved[-1]<0.9  ): ##0.8 # To allow more exploration if the Battacharyya distance between the estimated state and the target model is too high
            ##if (  len(self.B_saved)>0 and self.B_saved[-1]<0.96  ):
                random_nums[0] = np.random.normal(0, 1.2)
                random_nums[1] = np.random.normal(0, 1.2)
                #random_nums[6] = random_nums[6]/self.SCALE_CHANGE_D*0.01 # Equivalent to replacing self.SCALE_CHANGE_D by 0.01 if the Battacharyya distance between the estimated state and the target model is too high, so the size of the rectangle will not vary too much if the object is not seen well
                ###random_nums[6] = random_nums[6]/self.SCALE_CHANGE_D*0.001 # Equivalent to replacing self.SCALE_CHANGE_D by 0.001 if the Battacharyya distance between the estimated state and the target model is too high, so the size of the rectangle will not vary too much if the object is not seen well
                ###random_nums[6]=0
            '''    
            particle.x = int(particle.x+particle.x_dot*self.DELTA_T+random_nums[0]*particle.h_x+0.5)
            particle.y = int(particle.y+particle.y_dot*self.DELTA_T+random_nums[1]*particle.h_y+0.5)
            particle.x_dot = particle.x_dot+random_nums[2]*self.VELOCITY_DISTURB
            particle.y_dot = particle.y_dot+random_nums[3]*self.VELOCITY_DISTURB
            particle.h_x = int(particle.h_x*(particle.a_dot+1)+random_nums[4]*self.SCALE_DISTURB+0.5)
            particle.h_y = int(particle.h_y*(particle.a_dot+1)+random_nums[5]*self.SCALE_DISTURB+0.5)
            particle.a_dot = particle.a_dot+random_nums[6]*self.SCALE_CHANGE_D
            particle.draw_dot(self.img, self.out_path+'/%04d.jpg'%(self.img_index+1))

    def observe_single(self, i): #observe function for a single particle
        
        if self.particles[i].x<0 or self.particles[i].x>self.img.shape[1]-1 or self.particles[i].y<0 or self.particles[i].y>self.img.shape[0]-1:
            B_i=0
        else:
            p = [hist(num=self.H_num,max_range=180),hist(num=self.S_num,max_range=255),hist(num=self.V_num,max_range=255)]
            for hist_c in p:
                for u in range(hist_c.num):
                    a = np.sqrt(self.particles[i].h_x ** 2 + self.particles[i].h_y ** 2)
                    weight = []
                    x_bin = []
                    for m in range(self.particles[i].x - self.particles[i].h_x, self.particles[i].x + self.particles[i].h_x):
                        for n in range(self.particles[i].y - self.particles[i].h_y, self.particles[i].y + self.particles[i].h_y):
                            if n>=self.img.shape[0]:
                                n=self.imgHSV.shape[0]-1
                            elif n<0:
                                n=0
                            if m>=self.img.shape[1]:
                                m = self.imgHSV.shape[1] - 1
                            elif m<0:
                                m=0
                            x_val = self.imgHSV[n][m][p.index(hist_c)]
                            a1 = m - self.particles[i].x
                            a2 = n - self.particles[i].y
                            temp = k(np.sqrt(a1*a1 + a2*a2) / a)                            
                            x_bin.append(k_delta(hist_c.get_hist_id(x_val) - u))
                            weight.append(temp)                   
                    hist_c.height[u] = np.sum(np.array(weight) * np.array(x_bin))
            #print('i: ', i)
            #print('self.p[0].height: ', self.p[0].height)
            #print('self.p[1].height: ', self.p[1].height)
            #print('self.p[2].height: ', self.p[2].height)
            concatenated_hist_p = np.concatenate((p[0].height,p[1].height,p[2].height))
            f_normalization = np.sum(concatenated_hist_p)
            #print('f_normalization: ', f_normalization)
            concatenated_hist_p = concatenated_hist_p / f_normalization
            #print('concatenated_hist_p: ', concatenated_hist_p)
            B_i=B_coefficient(concatenated_hist_p, self.concatenated_hist_q)
            #print('B_i:', B_i)

        weights_i = get_weight(B_i)
        #print('get_weight(B_i): ', get_weight(B_i))
        return weights_i
        


    def observe(self):
        
        results_text = open(self.out_path+'/results_text.txt', 'a')
        
        self.imgHSV = cv.cvtColor(cv.imread(self.imgs[self.img_index]) , cv.COLOR_BGR2HSV)

        L = [i for i in range(self.particles_num)]
        
        #print('weights before processing observe', self.weights)
        
        if len(L)>processors_num:
            chunksize = len(L)//processors_num
        else:
            chunksize = 1 # if number of task inferior to number of processors then one task by processor
        if __name__ == '__main__':
            pool = multiprocessing.Pool(processes = processors_num)
            #seems to pass a lot of time here sometimes?
            weights = pool.map(self.observe_single, L, chunksize)
            #weights = p.map(self.observe_single, L)
            pool.close()
            pool.join()
            
            for i in range(len(weights)):
                self.weights[i] = weights[i]
        
        #print('weights ', weights) #parfois fait bugger l'ordi? (mais pas hydra)
        print('weights after processing ', self.weights)
        
        self.weights_before_normalization = self.weights
        weights_sum = sum(self.weights)
        for i in range(self.particles_num):
            self.weights[i]=self.weights[i]/weights_sum
            
        print('weights after normalization', self.weights)
        results_text.write('\nweights after normalization'+str(self.weights))

        '''for i in range(self.particles_num):
            print('dot: (%d,%d)  weight: %s'%(self.particles[i].x,self.particles[i].y,self.weights[i]))'''

    
    
    def mean_estimate(self): # Estimate the actual state by computing the mean for all particles considering their weights
    
        results_text = open(self.out_path+'/results_text.txt', 'a')
    
        self.state.x = np.sum(np.array([s.x for s in self.particles])*self.weights).astype(int)
        self.state.y = np.sum(np.array([s.y for s in self.particles])*self.weights).astype(int)
        self.state.h_x = np.sum(np.array([s.h_x for s in self.particles])*self.weights).astype(int)
        self.state.h_y = np.sum(np.array([s.h_y for s in self.particles])*self.weights).astype(int)
        self.state.x_dot = np.sum(np.array([s.x_dot for s in self.particles])*self.weights)
        self.state.y_dot = np.sum(np.array([s.y_dot for s in self.particles])*self.weights)
        self.state.a_dot = np.sum(np.array([s.a_dot for s in self.particles])*self.weights)
        print('img: %s  x: %s  y: %s  h_x: %s  h_y: %s  x_dot: %s  y_dot: %s  a_dot: %s'%(self.img_index+1,self.state.x,self.state.y,self.state.h_x,self.state.h_y,self.state.x_dot,self.state.y_dot,self.state.a_dot))
        results_text.write('\nimg: %s  x: %s  y: %s  h_x: %s  h_y: %s  x_dot: %s  y_dot: %s  a_dot: %s'%(self.img_index+1,self.state.x,self.state.y,self.state.h_x,self.state.h_y,self.state.x_dot,self.state.y_dot,self.state.a_dot))
        self.state.draw_rectangle(self.img,self.out_path+'/%04d.jpg'%(self.img_index+1))
      
    def cluster_estimate(self):
        
        results_text = open(self.out_path+'/results_text.txt', 'a')
              
        X=[] # data to give to the cluster algo
        index_highly_weighted_particles = []
        weights_highly_weighted_particles = []
        for i in range(self.particles_num):
            if self.weights[i]>(1/self.particles_num): ###  seuil de filtration des particles à faible poids à changer ? ********************
                X.append([self.particles[i].x, self.particles[i].y])
                index_highly_weighted_particles.append(i)
                weights_highly_weighted_particles.append(self.weights[i])
        n_highly_weighted_particles = len(index_highly_weighted_particles)
        print('Number of highly weighted particles: ', n_highly_weighted_particles)
        results_text.write('\nNumber of highly weighted particles: '+str( n_highly_weighted_particles))
        
        X=np.array(X) ### eps et min_samples à changer ? *************************
        print('Clustering...')  
        results_text.write('\nClustering...')
        clustering = DBSCAN(eps=max(self.state.h_x, self.state.h_y), min_samples=self.min_obs_in_clusters, n_jobs=-1).fit(X) # Density-Based Spatial Clustering of Applications with Noise (DBSCAN algo). n_jobs=-1 in order to use all CPU processors for routines that are parallelized with joblib.
        labels = clustering.labels_ # array of the labels of the clusters for each point of X, dtype=int64
        clusters_num = len(set(labels)) - (1 if -1 in labels else 0) # number of clusters (label -1 corresponding to noisy points that are not in any of the clusters)
        self.clusters_num = clusters_num
        print('Estimated number of clusters: ', clusters_num)
        results_text.write('\nEstimated number of clusters: '+str(clusters_num))
        b=[255/clusters_num*n for n in range(clusters_num)]
        
        if clusters_num>0:
            
            self.states_possible = []
            
            for n in range(clusters_num):
                
                state_x = 0
                state_y = 0
                state_h_x = 0
                state_h_y = 0
                state_x_dot = 0
                state_y_dot = 0
                state_a_dot = 0     
                
                sum_weights_current_cluster = 0
                
                for i in range(len(X)):
                    if labels[i]==n:
                        state_x += self.particles[index_highly_weighted_particles[i]].x*self.weights[index_highly_weighted_particles[i]]
                        state_y += self.particles[index_highly_weighted_particles[i]].y*self.weights[index_highly_weighted_particles[i]]
                        state_h_x += self.particles[index_highly_weighted_particles[i]].h_x*self.weights[index_highly_weighted_particles[i]]
                        state_h_y += self.particles[index_highly_weighted_particles[i]].h_y*self.weights[index_highly_weighted_particles[i]]
                        state_x_dot += self.particles[index_highly_weighted_particles[i]].x_dot*self.weights[index_highly_weighted_particles[i]]
                        state_y_dot += self.particles[index_highly_weighted_particles[i]].y_dot*self.weights[index_highly_weighted_particles[i]]
                        state_a_dot += self.particles[index_highly_weighted_particles[i]].a_dot*self.weights[index_highly_weighted_particles[i]]   
                        sum_weights_current_cluster += self.weights[index_highly_weighted_particles[i]]
                state_x = (state_x/sum_weights_current_cluster).astype(int)
                state_y = (state_y/sum_weights_current_cluster).astype(int)
                state_h_x = (state_h_x/sum_weights_current_cluster).astype(int)
                state_h_y = (state_h_y/sum_weights_current_cluster).astype(int)
                state_x_dot = state_x_dot/sum_weights_current_cluster
                state_y_dot = state_y_dot/sum_weights_current_cluster
                state_a_dot = state_a_dot/sum_weights_current_cluster 
                
                self.states_possible.append(state(x=state_x, y=state_y, x_dot=state_x_dot, y_dot=state_y_dot, h_x=state_h_x, h_y=state_h_y, a_dot=state_a_dot))
                
                print('img: %s cluster: %s  x: %s  y: %s  h_x: %s  h_y: %s  x_dot: %s  y_dot: %s  a_dot: %s'%(self.img_index+1,n,state_x,state_y,state_h_x,state_h_y,state_x_dot,state_y_dot,state_a_dot))
                results_text.write('\nimg: %s cluster: %s  x: %s  y: %s  h_x: %s  h_y: %s  x_dot: %s  y_dot: %s  a_dot: %s'%(self.img_index+1,n,state_x,state_y,state_h_x,state_h_y,state_x_dot,state_y_dot,state_a_dot))
                color = (b[n],255,0)# color in BGR
                img = self.img
                path = self.out_path+'/%04d.jpg'%(self.img_index+1)
                cv.rectangle(img, (state_x - state_h_x,state_y - state_h_y), ( state_x + state_h_x,state_y + state_h_y),color,thickness=1)
                self.img=img
                cv.imwrite(path, self.img)
               
    
    def update_target_u(self, u, hist_c, n_cluster):
        a = np.sqrt(self.states_possible[n_cluster].h_x ** 2 + self.states_possible[n_cluster].h_y ** 2)
        weight = []
        x_bin = []
        for m in range(self.states_possible[n_cluster].x - self.states_possible[n_cluster].h_x, self.states_possible[n_cluster].x + self.states_possible[n_cluster].h_x):
            for n in range(self.states_possible[n_cluster].y - self.states_possible[n_cluster].h_y, self.states_possible[n_cluster].y + self.states_possible[n_cluster].h_y):
                if n>=self.img.shape[0]:
                    n=self.imgHSV.shape[0]-1
                elif n<0:
                    n=0
                if m>=self.img.shape[1]:
                    m = self.imgHSV.shape[1] - 1
                elif m<0:
                    m=0
                x_val = self.imgHSV[n][m][self.pE_possible[n_cluster].index(hist_c)]
                a1 = m - self.states_possible[n_cluster].x
                a2 = n - self.states_possible[n_cluster].y
                temp = k(np.sqrt(a1*a1 + a2*a2) / a)                        
                x_bin.append(k_delta(hist_c.get_hist_id(x_val) - u))
                weight.append(temp)
        hist_c_height_u = np.sum(np.array(weight) * np.array(x_bin))   
        return hist_c_height_u
    
    
    
    
    def update_target(self):
        
        results_text = open(self.out_path+'/results_text.txt', 'a')
        
        alpha = 0.1 # alpha=0.1    alpha weights the contribution of the mean state histogram pE[St]
        
        print('concatenated_hist_q before update: ', self.concatenated_hist_q)
        results_text.write('\nconcatenated_hist_q before update:'+str(self.concatenated_hist_q))
        
        Bq = B_coefficient(self.concatenated_hist_q,self.concatenated_hist_q) # Battacharyya coefficient (similarity measure) between the target model q and the target model q. This is the maximal similarity measure, in order to allow below a normalization       
        print('Bq :'+str(Bq))
        results_text.write('\nBq :'+str(Bq))
        
        get_weight_Bq = get_weight(Bq)
        print('get_weight(Bq): '+str(get_weight_Bq))
        results_text.write('\nget_weight(Bq) :'+str(get_weight_Bq))
        
        self.B_possible = []
        
        self.pE_possible = [] # List of the histograms (HSV distribution) of the possible estimated state, one element by cluster
        self.concatenated_hist_pE_possible = []
        
        for n_cluster in range(self.clusters_num):
            
            # Creation of the histogram (HSV distribution) of the estimated state corresponding to the cluster number n_cluster:
            self.pE_possible.append([hist(num=self.H_num,max_range=180),hist(num=self.S_num,max_range=255),hist(num=self.V_num,max_range=255)])
            self.imgHSV = cv.cvtColor(cv.imread(self.imgs[self.img_index]) , cv.COLOR_BGR2HSV)
            
            for hist_c in self.pE_possible[n_cluster]:
                list_u = [(u, hist_c, n_cluster) for u in range(hist_c.num)]
    
                if len(list_u)>processors_num:
                    chunksize = len(list_u)//processors_num
                else:
                    chunksize = 1 # if number of task inferior to number of processors then one task by processor
                if __name__ == '__main__':
                    p = multiprocessing.Pool(processes = processors_num)
                    result = p.starmap(self.update_target_u, list_u, chunksize) 
                    p.close()
                    p.join()
                    
                    for u in range(hist_c.num):
                        hist_c.height[u] = result[u]

            concatenated_hist_pE = np.concatenate((self.pE_possible[n_cluster][0].height, self.pE_possible[n_cluster][1].height, self.pE_possible[n_cluster][2].height))
            f_normalization = np.sum(concatenated_hist_pE)
            concatenated_hist_pE = concatenated_hist_pE / f_normalization
            
            self.concatenated_hist_pE_possible.append(concatenated_hist_pE)
            
            # Calculate the Battacharyya coefficient between the estimated state and the target model q 
            B = B_coefficient(concatenated_hist_pE, self.concatenated_hist_q)
            
            
            print('n_cluster :'+str(n_cluster)+' , B :'+str(B))
            results_text.write('\nn_cluster :'+str(n_cluster)+' , B :'+str(B))
            '''B_normalized = B/Bq
            print('n_cluster :'+str(n_cluster)+' , B_normalized :'+str(B_normalized))
            results_text.write('\nn_cluster :'+str(n_cluster)+' , B_normalized :'+str(B_normalized))'''
            
            print('n_cluster :'+str(n_cluster)+' , get_weight(B) :'+str(get_weight(B)))
            results_text.write('\nn_cluster :'+str(n_cluster)+' , get_weight(B) :'+str(get_weight(B)))

            ##self.B_possible.append(B_normalized)
            self.B_possible.append(B)

        B_estimated = 0 # B_estimated will be the max of the B_normalized considering the B_normalized of each cluster, so it is the B_normalized of the "right" (or most probable) cluster
        for n_cluster in range(self.clusters_num):
            if self.B_possible[n_cluster]>B_estimated:
                B_estimated = self.B_possible[n_cluster]
                self.right_cluster = n_cluster


        self.B_saved.append(B_estimated)
        #print(self.B_saved)
            
        self.img_Neff_B.append(B_estimated)
        
        self.img_Neff_B.append(get_weight(B_estimated))
        
        print(self.img_Neff_B)
        results_text.write('\nimg_Neff_B :'+str(self.img_Neff_B))

        
        print('img: %s right cluster: %s  x: %s  y: %s  h_x: %s  h_y: %s  x_dot: %s  y_dot: %s  a_dot: %s'%(self.img_index+1,self.right_cluster,self.states_possible[self.right_cluster].x,self.states_possible[self.right_cluster].y,self.states_possible[self.right_cluster].h_x,self.states_possible[self.right_cluster].h_y,self.states_possible[self.right_cluster].x_dot,self.states_possible[self.right_cluster].y_dot,self.states_possible[self.right_cluster].a_dot))
        results_text.write('\nimg: %s right cluster: %s  x: %s  y: %s  h_x: %s  h_y: %s  x_dot: %s  y_dot: %s  a_dot: %s'%(self.img_index+1,self.right_cluster,self.states_possible[self.right_cluster].x,self.states_possible[self.right_cluster].y,self.states_possible[self.right_cluster].h_x,self.states_possible[self.right_cluster].h_y,self.states_possible[self.right_cluster].x_dot,self.states_possible[self.right_cluster].y_dot,self.states_possible[self.right_cluster].a_dot))

        color = (255,255,255)# color in BGR        

        # Updating the histogram of the target q under condition if the Battacharyya coefficient between the estimated state and the target model q is higher than a threshold T
        if B_estimated > self.T: # if high Battacharyya coefficient then we can assume that the object is not lost nor occulted so we update the target model using the estimated state
            color = (0,0,0) # black rectangle to show that this hypothesis is highly probable and to show that here was the update
            for u in range(len(self.concatenated_hist_q)):
                self.concatenated_hist_q[u] = (1-alpha)*self.concatenated_hist_q[u] + alpha*self.concatenated_hist_pE_possible[self.right_cluster][u]                

        img = self.img
        path = self.out_path+'/%04d.jpg'%(self.img_index+1)
        cv.rectangle(img, (self.states_possible[self.right_cluster].x - self.states_possible[self.right_cluster].h_x,self.states_possible[self.right_cluster].y - self.states_possible[self.right_cluster].h_y), ( self.states_possible[self.right_cluster].x + self.states_possible[self.right_cluster].h_x,self.states_possible[self.right_cluster].y + self.states_possible[self.right_cluster].h_y),color,thickness=1)
        self.img=img
        cv.imwrite(path, self.img)

        print('concatenated_hist_q after update: ', self.concatenated_hist_q)
        results_text.write('\nconcatenated_hist_q after update:'+str(self.concatenated_hist_q))
        
        if self.clusters_num > 0:
            print('concatenated_hist_pE_possible[self.right_cluster]: ', self.concatenated_hist_pE_possible[self.right_cluster])       
            results_text.write('\nconcatenated_hist_pE_possible[right_cluster]:'+str(self.concatenated_hist_pE_possible[self.right_cluster]))


        
'''        
        # Updating the histogram of the target q under condition if the Battacharyya coefficient between the estimated state and the target model q is higher than a threshold T
        if B_estimated > self.T: # if high Battacharyya coefficient then we can assume that the object is not lost nor occulted so we update the target model using the estimated state
            for i in range(len(self.q)):
                for u in range(self.q[i].num):
                    self.q[i].height[u] = (1-alpha)*self.q[i].height[u] + alpha*self.pE_possible[self.right_cluster][i].height[u]
'''








def main():
    
    tini = process_time()
    tini_counter=perf_counter()
    img_path = r'./test'
    out_path = r'./6Nef_alpha0_1_D0_001_proprand6var0_8_var0_4_proprand01var0_4_var2_p16_all_img_Bnorm_epsmaxh_min1_T0_97'
    
    results_text = open(out_path+'/results_text.txt', 'a')
    
    PF = ParticleFilter(particles_num=16, img_path=img_path, out_path=out_path, resampling_condition=6, min_obs_in_clusters = 1)    #particles_num = 56 #224 #20 or #200 #a multiple of the number of processors is a good choice like 224 for hydra
    
    t1 = process_time()
    t1_counter = perf_counter()
    duration_init = t1-tini
    duration_init_counter = t1_counter-tini_counter
    #print('init duration: '+str(duration_init))
    print('init duration_counter: '+str(duration_init_counter))
    results_text.write('init duration_counter: '+str(duration_init_counter))
    print()
    
    select_duration_list=[]
    propagate_duration_list=[]
    observe_duration_list=[]
    observe_duration_list_counter=[]
    estimate_duration_list=[]
    update_duration_list=[]
    update_duration_list_counter=[]
    resampling_duration_list=[]
    resampling_duration_list_counter=[]
    
    total_duration_list=[]
    total_duration_list_counter=[]
        
    #print(PF.imgs)
    
    while PF.img_index<(len(PF.imgs)-1):
    #while PF.img_index<360:
        
        print('\nImage: '+str(PF.img_index))
        
        tini_img=process_time()
        tini_img_counter=perf_counter()
        
        t0 = process_time()
        #PF.select()
        t1 = process_time()
        duration = t1-t0
        print('select duration: '+str(duration))
        results_text.write('\nselect duration: '+str(duration))
        select_duration_list.append(duration)
        select_duration_mean = np.sum(select_duration_list)/len(select_duration_list)
        print("select_duration_mean: ", select_duration_mean)
        results_text.write('\nselect duration mean: '+str(select_duration_mean))
        
        t0 = process_time()
        t0_counter = perf_counter()
        PF.resampling()
        t1 = process_time()
        t1_counter = perf_counter()
        duration = t1-t0
        duration_counter = t1_counter-t0_counter
        #print('resampling duration :'+str(duration))
        print('\nresampling duration counter:'+str(duration_counter))
        results_text.write('\nresampling duration counter:'+str(duration_counter))
        resampling_duration_list.append(duration)
        resampling_duration_list_counter.append(duration_counter)
        resampling_duration_mean = np.sum(resampling_duration_list)/len(resampling_duration_list)
        resampling_duration_mean_counter = np.sum(resampling_duration_list_counter)/len(resampling_duration_list_counter)
        #print("resampling_duration_mean :", resampling_duration_mean)
        print("resampling_duration_mean_counter :", resampling_duration_mean_counter)        
        results_text.write('\nresampling duration mean counter:'+str(resampling_duration_mean_counter))
        
        
        
        t0 = process_time()
        PF.propagate()
        t1 = process_time()
        duration = t1-t0
        print('propagate duration :'+str(duration))
        results_text.write('\npropagate duration :'+str(duration))
        propagate_duration_list.append(duration)
        propagate_duration_mean = np.sum(propagate_duration_list)/len(propagate_duration_list)
        print("propagate_duration_mean :", propagate_duration_mean)
        results_text.write('\npropagate_duration_mean:'+str(propagate_duration_mean))

        
        t0 = process_time()
        t0_counter = perf_counter()
        PF.observe()
        t1 = process_time()
        t1_counter = perf_counter()
        duration = t1-t0
        duration_counter = t1_counter-t0_counter
        #print('observe duration :'+str(duration))
        print('observe duration counter:'+str(duration_counter))
        results_text.write('\nobserve duration counter:'+str(duration_counter))
        observe_duration_list.append(duration)
        observe_duration_list_counter.append(duration_counter)
        observe_duration_mean = np.sum(observe_duration_list)/len(observe_duration_list)
        observe_duration_mean_counter = np.sum(observe_duration_list_counter)/len(observe_duration_list_counter)
        #print("observe_duration_mean :", observe_duration_mean)
        print("observe_duration_mean_counter :", observe_duration_mean_counter)
        results_text.write('\nobserve duration mean counter: '+str(observe_duration_mean_counter))

        
        t0 = process_time()
        PF.mean_estimate()
        t1 = process_time()
        duration = t1-t0
        print('estimate duration :'+str(duration))
        results_text.write('\nestimate duration :'+str(duration))
        estimate_duration_list.append(duration)
        estimate_duration_mean = np.sum(estimate_duration_list)/len(estimate_duration_list)
        print("estimate_duration_mean :", estimate_duration_mean)     
        results_text.write('\nestimate duration mean :'+str(estimate_duration_mean))

        
        PF.cluster_estimate()
        
        t0 = process_time()
        t0_counter = perf_counter()
        PF.update_target()
        t1 = process_time()
        t1_counter = perf_counter()
        duration = t1-t0
        duration_counter = t1_counter-t0_counter
        #print('update duration :'+str(duration))
        print('update duration counter:'+str(duration_counter))
        results_text.write('\nupdate duration counter:'+str(duration_counter))
        update_duration_list.append(duration)
        update_duration_list_counter.append(duration_counter)
        update_duration_mean = np.sum(update_duration_list)/len(update_duration_list)
        update_duration_mean_counter = np.sum(update_duration_list_counter)/len(update_duration_list_counter)
        #print("update_duration_mean :", update_duration_mean)
        print("update_duration_mean_counter :", update_duration_mean_counter)
        results_text.write('\nupdate mean duration counter:'+str(update_duration_mean_counter))

        
        tfin_img=process_time()
        tfin_img_counter=perf_counter()
        img_duration = tfin_img-tini_img
        img_duration_counter = tfin_img_counter-tini_img_counter
        #print('img_duration :', img_duration)
        print('img_duration_counter :', img_duration_counter)
        results_text.write('\nimg_duration_counter:'+str(img_duration_counter))
        total_duration_list.append(img_duration)
        total_duration_list_counter.append(img_duration_counter)
        img_duration_mean = np.sum(total_duration_list)/len(total_duration_list)
        img_duration_mean_counter = np.sum(total_duration_list_counter)/len(total_duration_list_counter)
        #print('img_duration_mean :', img_duration_mean)
        print('img_duration_mean_counter :', img_duration_mean_counter)
        results_text.write('\nimg_duration_mean_counter:'+str(img_duration_mean_counter))
        
        cumulative_duration = duration_init + sum(total_duration_list)
        cumulative_duration_counter = duration_init_counter + sum(total_duration_list_counter)
        #print('cumulative_duration :', cumulative_duration)
        print('cumulative_duration_counter :', cumulative_duration_counter)
        results_text.write('\ncumulative_duration_counter:'+str(cumulative_duration_counter))

        
    results_text.close()

if __name__=='__main__':
    main()
    
'''
import cProfile
import re
cProfile.run('main()')
'''
