######################################################################
######################################################################
##########       This file defines the model class          ########## 
######################################################################
######################################################################



import numpy as np
import tensorflow as tf
import time
import logging
from tensorflow import keras
import json 
import pathlib
import matplotlib.pyplot as plt
import os 
import pandas as pd
from IntraTemporalSolver import *

tf.random.set_seed(20001031)


class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, config):
        super(FeedForwardSubNet, self).__init__(name = config["nn_name"] + ".init_layer")
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5),
                name = config["nn_name"] + ".bn." + str(_)
            )
            for _ in range(len(config["num_hiddens"]) + 1)]
        
        if config['activation'] is not None and "relu" in config['activation']:
            initializer = tf.keras.initializers.HeNormal(seed=0)
        else:
            initializer = tf.keras.initializers.GlorotUniform(seed=0)

        self.dense_layers = [tf.keras.layers.Dense(config["num_hiddens"][i],
                                                   use_bias=config['use_bias'],
                                                   activation=config['activation'],
                                                   kernel_initializer = initializer,
                                                   name = config["nn_name"] + ".dense." + str(i))
                             for i in range(len(config["num_hiddens"]))]
        # final output should be gradient of size dim
        try:
            if config['final_activation'] is None:
                initializer = tf.keras.initializers.GlorotUniform(seed=0)
            elif "relu" in config['final_activation']:
                initializer = tf.keras.initializers.HeNormal(seed=0)
            else:
                initializer = tf.keras.initializers.GlorotUniform(seed=0)
        except:
            initializer = tf.keras.initializers.GlorotUniform(seed=0)

        self.dense_layers.append(tf.keras.layers.Dense(config["dim"], 
        kernel_initializer = initializer, 
        activation=config['final_activation'], use_bias = True, name = config["nn_name"] + ".output" ))

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        x = self.bn_layers[0](x, training)
        x_inputs = []
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            x_inputs.append(x)
        x = tf.keras.layers.Add()(x_inputs)
        x = self.dense_layers[-1](x)
        return x
    
class model:
    def __init__(self, params):
        
        ## Load parameters
        self.params  = params 
        
        ## Table 1: Economic Parameters
        self.params["δ"] = 0.02 # subjective discount rate
        self.params["ρ"] = 1.0 # Intertemporal Elasticity of Substitution
        
        self.params["α"] =  0.5 # AI sector production function
        self.params["θ"] =  0.5 # Artificial Labor
        self.params['γ'] =  0.1 # Elasticity of Substitution between Human labor and Artificial Labor
        self.params['ι'] =  0.5 # Proportion of human labor 
        self.params['β'] = 0.3 
        self.params["ψ"] = 1
        
        
        ## Underlying dynamics
        self.params["μ_Z"] =  0.04; self.params["σ_Z"] = 0.035
        self.params["μ_a"] =  0.05 ; self.params["κ_a"] =  6;   self.params["σ_a"] = 0.01
        self.params["μ_g"] =  0.035 ; self.params["κ_g"] = 7;   self.params["σ_g"] = 0.01 
        self.params["ζ"] = 0.0; self.params["ψ_0"] = 0.10  ;self.params["ψ_1"] = 0.5; self.params["σ_κ"] = 0.0078
         
        self.params["A_g"] =  0.12 
        
           
        ## Table 3: State Variable Initial Values and Ranges
        self.params["K_g_0"] = 10.0
        self.params["K_a_0"] = 5.0
        self.params["Z_0"] = 5
        self.params["D_0"] = 5 
        
        self.params["logK_g_min"] = 0.01
        self.params["logK_g_max"] = 4.0
        self.params["logK_a_min"] = 0.01
        self.params["logK_a_max"] = 2.0
        self.params["logZ_min"] = 0.01
        self.params["logZ_max"] = 2.0
        self.params["logD_min"] = 0.01
        self.params["logD_max"] = 2.0 
        
        
        if 'tensorboard' not in params.keys():
            print("Tensorboard option not detected; setting to False by default.")
            self.params['tensorboard'] = False 
        print("Tensorboard boolean =", self.params['tensorboard'] )


        ## Create neural networks
        self.v_nn    = FeedForwardSubNet(self.params['v_nn_config'])
        self.i_g_nn  = FeedForwardSubNet(self.params['i_g_nn_config'])
        self.i_a_nn  = FeedForwardSubNet(self.params['i_a_nn_config'])
        self.i_d_nn  = FeedForwardSubNet(self.params['i_d_nn_config'])
        
        ## firm prices
        self.v_g_nn    = FeedForwardSubNet(self.params['v_g_nn_config'])
        self.v_a_nn    = FeedForwardSubNet(self.params['v_a_nn_config'])
        ## Create folder 
        pathlib.Path(self.params["export_folder"]).mkdir(parents=True, exist_ok=True) 

        ## Create ranges for sampling later 
        self.params["state_intervals"] = {}
 
        self.params["state_intervals"]["logK_g"]     =  tf.reshape(tf.linspace(self.params['logK_g_min'], self.params['logK_g_max'], self.params['batch_size'] + 1), (self.params['batch_size'] + 1,1))
        self.params["state_intervals"]["logK_g_interval_size"] =  self.params["state_intervals"]["logK_g"][1] -  self.params["state_intervals"]["logK_g"][0]

        self.params["state_intervals"]["logK_a"]     =  tf.reshape(tf.linspace(self.params['logK_a_min'], self.params['logK_a_max'], self.params['batch_size'] + 1), (self.params['batch_size'] + 1,1))
        self.params["state_intervals"]["logK_a_interval_size"] =  self.params["state_intervals"]["logK_a"][1] -  self.params["state_intervals"]["logK_a"][0]

        self.params["state_intervals"]["logZ"]        =  tf.reshape(tf.linspace(self.params['logZ_min'], self.params['logZ_max'], self.params['batch_size'] + 1), (self.params['batch_size'] + 1,1))
        self.params["state_intervals"]["logZ_interval_size"] =  self.params["state_intervals"]["logZ"][1] -  self.params["state_intervals"]["logZ"][0]
 
        self.params["state_intervals"]["logD"]        =  tf.reshape(tf.linspace(self.params['logD_min'], self.params['logD_max'], self.params['batch_size'] + 1), (self.params['batch_size'] + 1,1))
        self.params["state_intervals"]["logD_interval_size"] =  self.params["state_intervals"]["logD"][1] -  self.params["state_intervals"]["logD"][0]



        ## Create objects to generate checkpoints for tensorboard
        pathlib.Path(self.params["export_folder"] + '/logs/train/').mkdir(parents=True, exist_ok=True) 
        pathlib.Path(self.params["export_folder"] + '/logs/test/').mkdir(parents=True, exist_ok=True) 
        self.train_writer = tf.summary.create_file_writer( self.params["export_folder"] + '/logs/train/')
        self.test_writer  = tf.summary.create_file_writer( self.params["export_folder"] + '/logs/test/')


    def sample(self):
        '''
        Sampling all state variables. Not all variables are used in Calculation. 
        '''
        
        offsets      = tf.random.uniform(shape=(self.params['batch_size'],1), minval=0.0, maxval=1.0)
        logK_g         = tf.random.shuffle(self.params["state_intervals"]["logK_g"][:-1] + self.params["state_intervals"]["logK_g_interval_size"] * offsets)

        offsets      = tf.random.uniform(shape=(self.params['batch_size'],1), minval=0.0, maxval=1.0)
        logK_a         = tf.random.shuffle(self.params["state_intervals"]["logK_a"][:-1] + self.params["state_intervals"]["logK_a_interval_size"] * offsets)
        
        
        offsets      = tf.random.uniform(shape=(self.params['batch_size'],1), minval=0.0, maxval=1.0)
        logZ            = tf.random.shuffle(self.params["state_intervals"]["logZ"][:-1] + self.params["state_intervals"]["logZ_interval_size"] * offsets)

 
        
        offsets            = tf.random.uniform(shape=(self.params['batch_size'],1), minval=0.0, maxval=1.0)
        logD            = tf.random.shuffle(self.params["state_intervals"]["logD"][:-1] + self.params["state_intervals"]["logD_interval_size"] * offsets)
        
        L_g, L_a, w, p = solve_IntraTemporal_batch_tf(logK_g, logK_a, logZ, logD, self.params)
        
        return logK_g, logK_a, logZ,logD,L_g, L_a, w, p
    
    
 

    @tf.function
    def pde_rhs(self, logK_g, logK_a, logZ,logD,L_g, L_a, w, p):
        '''
        This is the RHS of the HJB equation
        '''
 
        δ = self.params["δ"]
        ρ = self.params["ρ"]
        α = self.params["α"]
        θ = self.params["θ"]
        γ = self.params["γ"]
        ι = self.params["ι"]
        β = self.params["β"]
        ψ = self.params["ψ"]  

        μ_Z = self.params["μ_Z"]
        σ_Z = self.params["σ_Z"]

        μ_a = self.params["μ_a"]
        κ_a = self.params["κ_a"]
        σ_a = self.params["σ_a"]

        μ_g = self.params["μ_g"]
        κ_g = self.params["κ_g"]
        σ_g = self.params["σ_g"]
        ζ = self.params["ζ"]
        ψ_0 = self.params["ψ_0"]
        ψ_1 = self.params["ψ_1"]
        σ_κ = self.params["σ_κ"]

        A_g = self.params["A_g"]


        
         
        state = tf.concat([logK_g, logK_a , logZ,logD], 1) 
            
 
        
        ## Evalute neural networks 
        v            = self.v_nn(state)
        i_g          = self.i_g_nn(state)
        i_a          = self.i_a_nn(state)
        i_d          = self.i_d_nn(state)
        
        
        S_g            = self.v_g_nn(state)
        S_a            = self.v_a_nn(state)
         

        ## Calculate some variables for proceeding calculation. 

        K_a = tf.reshape(tf.exp(logK_a), [self.params['batch_size'], 1])
        K_g = tf.reshape(tf.exp(logK_g), [self.params['batch_size'], 1])
        Z = tf.reshape(tf.exp(logZ), [self.params['batch_size'], 1])
        D = tf.reshape(tf.exp(logD), [self.params['batch_size'], 1])

        #########################
        #### Calculate Partial Derivatives
        #########################
        
        ## FOC w.r.t logK_a
        dv_dlogK_a                 = tf.reshape(tf.gradients(v, logK_a, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        dv_ddlogK_a                = tf.reshape(tf.gradients(dv_dlogK_a, logK_a, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        
        dS_g_dlogK_a                 = tf.reshape(tf.gradients(S_g, logK_a, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        dS_g_ddlogK_a                = tf.reshape(tf.gradients(dS_g_dlogK_a, logK_a, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        
        dS_a_dlogK_a                 = tf.reshape(tf.gradients(S_g, logK_a, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        dS_a_ddlogK_a                = tf.reshape(tf.gradients(dS_a_dlogK_a, logK_a, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        
         
        ## FOC w.r.t logK_g
        dv_dlogK_g                 = tf.reshape(tf.gradients(v, logK_g, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        dv_ddlogK_g                = tf.reshape(tf.gradients(dv_dlogK_g, logK_g, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        
        dS_g_dlogK_g                 = tf.reshape(tf.gradients(S_g, logK_g, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        dS_g_ddlogK_g                = tf.reshape(tf.gradients(dS_g_dlogK_g, logK_g, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])    
        
        dS_a_dlogK_g                 = tf.reshape(tf.gradients(S_a, logK_g, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        dS_a_ddlogK_g                = tf.reshape(tf.gradients(dS_a_dlogK_g, logK_g, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])    
        
        
        ## FOC w.r.t logZ
        dv_dlogZ                    = tf.reshape(tf.gradients(v, logZ, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        dv_ddlogZ                   = tf.reshape(tf.gradients(dv_dlogZ, logZ, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])

        dS_g_dlogZ                    = tf.reshape(tf.gradients(S_g, logZ, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        dS_g_ddlogZ                   = tf.reshape(tf.gradients(dS_g_dlogZ, logZ, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])

        dS_a_dlogZ                    = tf.reshape(tf.gradients(S_a, logZ, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        dS_a_ddlogZ                   = tf.reshape(tf.gradients(dS_a_dlogZ, logZ, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])


        ## FOC w.r.t logD
        dv_dlogD                 = tf.reshape(tf.gradients(v, logD, unconnected_gradients='zero')[0], [self.params["batch_size"], 1])
        dv_ddlogD                 = tf.reshape(tf.gradients(dv_dlogD, logD, unconnected_gradients='zero')[0], [self.params["batch_size"], 1])
 
        dS_g_dlogD                 = tf.reshape(tf.gradients(S_g, logD, unconnected_gradients='zero')[0], [self.params["batch_size"], 1])
        dS_g_ddlogD                 = tf.reshape(tf.gradients(dS_g_dlogD, logD, unconnected_gradients='zero')[0], [self.params["batch_size"], 1])
 
        dS_a_dlogD                 = tf.reshape(tf.gradients(S_a, logD, unconnected_gradients='zero')[0], [self.params["batch_size"], 1])
        dS_a_ddlogD                 = tf.reshape(tf.gradients(dS_a_dlogD, logD, unconnected_gradients='zero')[0], [self.params["batch_size"], 1])
 
 
 
        #########################################################
        ######## RHS without Jump terms #########################
        #########################################################
 
        N = 1 - L_a - L_g
        X = Z * ( K_a**α ) * ( L_a**(1-α) )
        
        L_AI =  (D**θ) * (X**(1-θ))
        
        y = A_g * (K_g**β) * (ι* L_g**γ + (1-ι) *L_AI**γ)**((1-β)/γ)

        C = y - i_g*K_g - i_a * K_a - i_d * K_g
        
 
        inside_log   =   tf.reshape( tf.math.maximum( C , 0.000000001), [self.params["batch_size"], 1])
        
        ## Risk free rate
        rf = δ + 0.5 * (y/inside_log) * β**2 *  σ_g**2 + 0.5 * (y/inside_log) * (1-β) * (1-ι) *( L_AI / (ι* L_g**γ + (1-ι) *L_AI**γ)**(1/γ)   )**γ *(
         ι*(  L_g / (ι* L_g**γ + (1-ι) *L_AI**γ)**(1/γ)   )**γ + (1-β) * (1-ι) )*(
            ρ**2 * σ_κ**2 + α**2 * γ**2 * (1-ρ)**2 * σ_a**2 + γ**2 * (1-ρ)**2 * σ_Z**2
        )
         
        sdf = δ /(inside_log* N**ψ )
        ## Profit of two sectors
        Π_g = y - w * L_g -p*X
        Π_a =  p*X -w * L_a
        
        ## Risk Price
        λ_g = β * (y/inside_log) * σ_g 
        λ_a = α* γ * (1-ρ)*(1-ι)*(1-β) * (y/inside_log) *( L_AI / (ι* L_g**γ + (1-ι) *L_AI**γ)**(1/γ)   )**γ * σ_a
        λ_D = ρ*(1-ι)*(1-β) * (y/inside_log) * ( L_AI / (ι* L_g**γ + (1-ι) *L_AI**γ)**(1/γ)   )**γ * σ_κ
        λ_Z = (1-ρ)*(1-ι)*(1-β) * (y/inside_log) * ( L_AI / (ι* L_g**γ + (1-ι) *L_AI**γ)**(1/γ)   )**γ * σ_Z
        
        # flow           = δ /(1-ρ) *  (  ( inside_log * N**ψ /tf.exp(v) )**(1-ρ)   -1  )
        
        flow = δ* tf.math.log(  inside_log ) + ψ * tf.math.log( N )
        
        v_k_g_term       =  - μ_g + i_g - κ_g/2 * (i_g)**2 - σ_g**2/2
        v_kk_g_term      =    0.5 * tf.pow(σ_g, 2)
        
        v_k_a_term       =  - μ_a + i_a - κ_a/2 * (i_a)**2 - σ_a**2/2
        v_kk_a_term      =    0.5 * tf.pow(σ_a, 2)
        
        

        v_logZ_term     =   -  μ_Z    - 0.5 * tf.pow(σ_Z, 2)
        v_logZlogZ_term =     0.5 * tf.pow(σ_Z, 2)
        
            
        v_logD_term     = -  ζ  +  ψ_0  * (K_g/D * i_d)**ψ_1 - 0.5 * tf.pow(σ_κ, 2)
        v_logDlogD_term = 0.5 * tf.pow(σ_κ, 2)
        

        rhs = flow  -  δ *  v \
                + v_k_g_term * dv_dlogK_g + v_kk_g_term * dv_ddlogK_g  \
                +v_k_a_term * dv_dlogK_a + v_kk_a_term * dv_ddlogK_a \
                +v_logZ_term * dv_dlogZ+ v_logZlogZ_term * dv_ddlogZ \
                + v_logD_term*dv_dlogD + v_logDlogD_term*dv_ddlogD
 
  

        rhs_S_g = sdf* Π_g  -rf  - λ_a* σ_a * dS_g_dlogK_a - λ_g* σ_g * dS_g_dlogK_g   -λ_D*σ_κ* dS_g_dlogD  -λ_Z*σ_Z*dS_g_ddlogZ \
                + v_k_g_term * dS_g_dlogK_g + v_kk_g_term * dS_g_ddlogK_g  \
                +v_k_a_term * dS_g_dlogK_a + v_kk_a_term * dS_g_ddlogK_a \
                +v_logZ_term * dS_g_dlogZ+ v_logZlogZ_term * dS_g_ddlogZ \
                + v_logD_term*dS_g_dlogD + v_logDlogD_term*dS_g_ddlogD
        
   
        rhs_S_a = Π_a -rf  - λ_a* σ_a * dS_a_dlogK_a - λ_g* σ_g * dS_a_dlogK_g   -λ_D*σ_κ* dS_a_dlogD  -λ_Z*σ_Z*dS_a_ddlogZ \
                + v_k_g_term * dS_a_dlogK_g + v_kk_g_term * dS_a_ddlogK_g  \
                +v_k_a_term * dS_a_dlogK_a + v_kk_a_term * dS_a_ddlogK_a \
                +v_logZ_term * dS_a_dlogZ+ v_logZlogZ_term * dS_a_ddlogZ \
                + v_logD_term*dS_a_dlogD + v_logDlogD_term*dS_a_ddlogD
        ###################################################
        ######### FOCs w.r.t controls lar
        ###################################################
    
        

        FOC_g   = - dv_dlogK_g*(1-κ_g * i_g)  + δ * K_g / inside_log
        FOC_a   = - dv_dlogK_a*(1-κ_a * i_a)  + δ * K_a / inside_log
        FOC_D   = - dv_dlogD*ψ_0  *  ψ_1* i_d**(ψ_1-1) * (K_g/D )**ψ_1 +  δ * K_g / inside_log
        
        return  rhs,FOC_g,FOC_a,FOC_D ,  C, rhs_S_g, rhs_S_a

    @tf.function
    def objective_fn(self, logK_g, logK_a , logZ,logD,L_g, L_a, w, p,  compute_control, training):
 
        ## This is the objective function that stochastic gradient descend will try to minimize
        ## It depends on which NN it is training. Controls and value functions have different
        ## objectives.

        rhs,FOC_g,FOC_a,FOC_D,  c , rhs_S_g, rhs_S_a    = self.pde_rhs(logK_g, logK_a , logZ,logD,L_g, L_a, w, p)

        epsilon = 10e-4
        negative_consumption_boolean = tf.reshape( tf.cast( c < 0.000000001, tf.float32 ),  [self.params["batch_size"], 1])
        loss_c  = - c  * negative_consumption_boolean + epsilon

 
        if training:    
            ## Take care of nonsensical controls first

            loss_c_mse = tf.sqrt(tf.reduce_mean(tf.square(loss_c  )))        
            
            control_constraints = tf.reduce_sum(negative_consumption_boolean)   
            loss_constraints    = loss_c_mse  

            if control_constraints > 0:
                return loss_constraints  


            if compute_control:
  
                return -tf.reduce_mean( (rhs   ) ) + \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_g )))  + \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_a )))   + \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_D )))   
                        
            else:
                loss = tf.sqrt(tf.reduce_mean(tf.square( (rhs ) ))) + \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_g )))  + \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_a )))   + \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_D )))   + \
                        tf.sqrt(tf.reduce_mean(tf.square(rhs_S_g )))   + \
                      tf.sqrt(tf.reduce_mean(tf.square(rhs_S_a )))    
                return loss  

        else:
            return tf.sqrt(tf.reduce_mean(tf.square( (rhs ) ))) , \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_g )))  , \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_a )))  , \
                        tf.sqrt(tf.reduce_mean(tf.square(FOC_D )))    ,\
                        tf.sqrt(tf.reduce_mean(tf.square(rhs_S_g ))) ,\
                        tf.sqrt(tf.reduce_mean(tf.square(rhs_S_a )))    

                             
                             
                             
    def grad(self, logK_g, logK_a , logZ,logD,L_g, L_a, w, p,  compute_control , training ):
        
        
        if compute_control:
            with tf.GradientTape(persistent=True) as tape:
                objective  = self.objective_fn(logK_g, logK_a , logZ,logD,L_g, L_a, w, p,  compute_control, training)

            trainable_variables = self.i_g_nn.trainable_variables + self.i_d_nn.trainable_variables + self.i_a_nn.trainable_variables  + self.v_g_nn.trainable_variables+self.v_a_nn.trainable_variables

            grad = tape.gradient(objective, trainable_variables)

            del tape

            return grad, objective 
        else:
            with tf.GradientTape(persistent=True) as tape:
                objective  = self.objective_fn(logK_g, logK_a , logZ,logD,L_g, L_a, w, p,  compute_control, training)
            
            grad = tape.gradient(objective, self.v_nn.trainable_variables )
            del tape

            return grad , objective 

    @tf.function
    def train_step(self):
        logK_g, logK_a , logZ,logD,L_g, L_a, w, p = self.sample()
  
        ## First, train value function
        
        grad, loss_v_train = self.grad(logK_g, logK_a , logZ,logD,L_g, L_a, w, p,  False,  True) # compute_control=, training=
        self.params["optimizers"][0].apply_gradients(zip(grad, self.v_nn.trainable_variables))

        ## Second, train controls
        grad, loss_c_train  = self.grad(logK_g, logK_a , logZ,logD,L_g, L_a, w, p,  True,  True)

        self.params["optimizers"][1].apply_gradients(zip(grad, self.i_g_nn.trainable_variables + self.i_d_nn.trainable_variables + self.i_a_nn.trainable_variables + self.v_g_nn.trainable_variables+self.v_a_nn.trainable_variables))

        return loss_v_train, loss_c_train 
    
    
    def train(self):

        start_time = time.time()
        training_history = []

        # Prepare to store best neural networks and initialize networks
        min_loss = float("inf")
        
        n_inputs = 4

        best_v_nn    = FeedForwardSubNet(self.params['v_nn_config'])
        best_v_nn.build( (self.params["batch_size"], n_inputs) ) 
        self.v_nn.build( (self.params["batch_size"], n_inputs) )

        best_i_g_nn  = FeedForwardSubNet(self.params['i_g_nn_config'])
        best_i_g_nn.build( (self.params["batch_size"], n_inputs) ) 
        self.i_g_nn.build( (self.params["batch_size"], n_inputs) )

        best_i_a_nn  = FeedForwardSubNet(self.params['i_a_nn_config'])
        best_i_a_nn.build( (self.params["batch_size"], n_inputs) ) 
        self.i_a_nn.build( (self.params["batch_size"], n_inputs) )

        best_i_d_nn  = FeedForwardSubNet(self.params['i_d_nn_config'])
        best_i_d_nn.build( (self.params["batch_size"], n_inputs) ) 
        self.i_d_nn.build( (self.params["batch_size"], n_inputs) )
        
        best_v_g_nn    = FeedForwardSubNet(self.params['v_g_nn_config'])
        best_v_g_nn.build( (self.params["batch_size"], n_inputs) ) 
        self.v_g_nn.build( (self.params["batch_size"], n_inputs) )

        best_v_a_nn    = FeedForwardSubNet(self.params['v_a_nn_config'])
        best_v_a_nn.build( (self.params["batch_size"], n_inputs) ) 
        self.v_a_nn.build( (self.params["batch_size"], n_inputs) )


        best_v_nn.set_weights(self.v_nn.get_weights())
        best_i_g_nn.set_weights(self.i_g_nn.get_weights())
        best_i_a_nn.set_weights(self.i_a_nn.get_weights())
        best_i_d_nn.set_weights(self.i_d_nn.get_weights())

        best_v_g_nn.set_weights(self.v_g_nn.get_weights())
        best_v_a_nn.set_weights(self.v_a_nn.get_weights())

        ## Load pretrained weights
        if self.params['pretrained_path'] is not None:
            self.v_nn.load_weights( self.params["pretrained_path"]  + '/v_nn_checkpoint')
            self.i_g_nn.load_weights( self.params["pretrained_path"]  + '/i_g_nn_checkpoint')
            self.i_d_nn.load_weights( self.params["pretrained_path"]  + '/i_d_nn_checkpoint')
            self.i_a_nn.load_weights( self.params["pretrained_path"]  + '/i_a_nn_checkpoint')



        # begin sgd iteration
        # begin sgd iteration
        for step in range(self.params["num_iterations"]):
            if step % self.params["logging_frequency"] == 0:
                ## Sample test data
                logK_g, logK_a , logZ,logD,L_g, L_a, w, p = self.sample()
                ## Compute test loss
                test_losses = self.objective_fn(logK_g, logK_a , logZ,logD,L_g, L_a, w, p, False, False)  #compute_control, training 

                ## Store best neural networks
                if (test_losses[0] < min_loss):
                    min_loss = test_losses[0]

                    best_v_nn.set_weights(self.v_nn.get_weights())
                    best_i_g_nn.set_weights(self.i_g_nn.get_weights())
                    best_i_a_nn.set_weights(self.i_a_nn.get_weights())
                    best_i_d_nn.set_weights(self.i_d_nn.get_weights())
                    best_v_a_nn.set_weights(self.v_a_nn.get_weights())
                    best_v_g_nn.set_weights(self.v_g_nn.get_weights())

                ## Generate checkpoints for tensorboard
                
                if self.params['tensorboard']:
                    grad_v_nn,loss_v_train = self.grad(logK_g, logK_a , logZ,logD,L_g, L_a, w, p,     False,  True)
                    grad_controls,loss_c_train = self.grad(logK_g, logK_a , logZ,logD,L_g, L_a, w, p,     True,  True)

                    with self.test_writer.as_default():
                        ## Export learning rates
                        for optimizer_idx in range(len(self.params['optimizers'])):
                            if "sgd" in self.params['learning_rate_schedule_type']:
                                tf.summary.scalar('learning_rate_' + str(optimizer_idx), self.params["optimizers"][optimizer_idx]._decayed_lr(tf.float32), step=step)
                            elif "piecewiseconstant" in self.params['learning_rate_schedule_type']:
                                optimizer = self.params["optimizers"][optimizer_idx]
                                current_lr = optimizer.learning_rate(step) if isinstance(optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule) else optimizer.lr
                                tf.summary.scalar(f'learning_rate_{optimizer_idx}', current_lr, step=step)
                            else:
                                tf.summary.scalar('learning_rate_' + str(optimizer_idx), self.params["optimizers"][optimizer_idx].lr, step=step)

                        
                        tf.summary.scalar('loss_value_function', test_losses[0], step=step)
                        tf.summary.scalar('loss_FOC_g', test_losses[1], step=step)
                        tf.summary.scalar('loss_FOC_a', test_losses[2], step=step)
                        tf.summary.scalar('loss_FOC_d', test_losses[3], step=step)
                        tf.summary.scalar('loss_FOC_S_g', test_losses[4], step=step)
                        tf.summary.scalar('loss_FOC_S_a', test_losses[5], step=step)
                     
                            
                            
                        ## Export weights and gradients
                        
                        #### v_nn
                        for layer in self.v_nn.layers:
                            for W in layer.weights:
                                tf.summary.histogram(W.name + '_weights', W, step=step)
                        for g in range(len(self.v_nn.trainable_variables)):
                            tf.summary.histogram(self.v_nn.trainable_variables[g].name + '_grads', grad_v_nn[g], step=step)

                        #### i_g
                        for layer in self.i_g_nn.layers:
                            for W in layer.weights:
                                tf.summary.histogram(W.name + '_weights', W, step=step)
                        for g in range(len(self.i_g_nn.trainable_variables)):
                            tf.summary.histogram(self.i_g_nn.trainable_variables[g].name + '_grads', grad_controls[g], step=step)

                        #### i_d
                        for layer in self.i_d_nn.layers:
                            for W in layer.weights:
                                tf.summary.histogram(W.name + '_weights', W, step=step)
                        for g in range(len(self.i_d_nn.trainable_variables)):
                            tf.summary.histogram(self.i_d_nn.trainable_variables[g].name + '_grads', grad_controls[len(self.i_g_nn.trainable_variables) + g], step=step)

                        ### i_a
                        for layer in self.i_a_nn.layers:
                            for W in layer.weights:
                                tf.summary.histogram(W.name + '_weights', W, step=step)
                        for g in range(len(self.i_a_nn.trainable_variables)):
                            tf.summary.histogram(self.i_a_nn.trainable_variables[g].name + '_grads', grad_controls[len(self.i_a_nn.trainable_variables) + g], step=step)
 
                
                elapsed_time = time.time() - start_time

                ## Appending to training history
                entry = [step] + list(test_losses) + [ elapsed_time]
                training_history.append(entry)

        

                ## Save training history
                header = 'step,loss_value_function,loss_FOC_g,loss_FOC_a,loss_FOC_d,loss_FOC_S_g,loss_FOC_S_a,elapsed_time'

                np.savetxt(self.params["export_folder"] + '/training_history.csv',
                        training_history,
                        fmt=['%d'] + ['%.5e'] * len(test_losses) + ['%d'],
                        delimiter=",",
                        header=header,
                        comments='')

            loss_v_train, loss_c_train  = self.train_step()
            if self.params['tensorboard'] and step % self.params["logging_frequency"] == 0:
                with self.train_writer.as_default():
                    tf.summary.scalar('loss_value_train', loss_v_train, step=step)
                    tf.summary.scalar('loss_control_train', loss_c_train, step=step) 


        ## Use best neural networks 
        self.v_nn.set_weights(best_v_nn.get_weights())
        self.i_g_nn.set_weights(best_i_g_nn.get_weights())
        self.i_a_nn.set_weights(best_i_a_nn.get_weights())
        self.i_d_nn.set_weights(best_i_d_nn.get_weights())
        self.v_a_nn.set_weights(best_v_a_nn.get_weights())
        self.v_g_nn.set_weights(best_v_g_nn.get_weights())


        ## Export last check point
        self.v_nn.save_weights( self.params["export_folder"] + '/v_nn_checkpoint')
        self.i_g_nn.save_weights( self.params["export_folder"] + '/i_g_nn_checkpoint')
        self.i_a_nn.save_weights( self.params["export_folder"] + '/i_a_nn_checkpoint')
        self.i_d_nn.save_weights( self.params["export_folder"] + '/i_d_nn_checkpoint' )
        self.v_a_nn.save_weights( self.params["export_folder"] + '/v_a_nn_checkpoint')
        self.v_g_nn.save_weights( self.params["export_folder"] + '/v_g_nn_checkpoint' )

 
        ## Save training history
        header = 'step,loss_value_function,loss_FOC_g,loss_FOC_a,loss_FOC_d,loss_FOC_S_g,loss_FOC_S_a,elapsed_time'

        np.savetxt(self.params["export_folder"] + '/training_history.csv',
                training_history,
                fmt=['%d'] + ['%.5e'] * len(test_losses) + ['%d'],
                delimiter=",",
                header=header,
                comments='')
        
        ## Plot losses
 
        loss_value_function                    = [history_record[1] for history_record in training_history]
        loss_FOC_g                             = [history_record[2] for history_record in training_history]
        loss_FOC_a                             = [history_record[3] for history_record in training_history]
        loss_FOC_d                             = [history_record[4] for history_record in training_history]
        loss_FOC_S_g                           = [history_record[5] for history_record in training_history]
        loss_FOC_S_a                           = [history_record[6] for history_record in training_history]
  
        plt.figure()
        plt.title("loss_value_function")
        plt.plot(loss_value_function)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_value_function.png")
        plt.close()


        plt.figure()
        plt.title("loss_FOC_g")
        plt.plot(loss_FOC_g)
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_FOC_g.png")
        plt.close()

        plt.figure()
        plt.title("loss_FOC_a")
        plt.plot(loss_FOC_a)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_FOC_a.png")
        plt.close()

        plt.figure()
        plt.title("loss_FOC_d")
        plt.plot(loss_FOC_d)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_FOC_d.png")
        plt.close()

        plt.figure()
        plt.title("loss_FOC_S_g")
        plt.plot(loss_FOC_S_g)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_FOC_S_g.png")
        plt.close()

        plt.figure()
        plt.title("loss_FOC_S_a")
        plt.plot(loss_FOC_S_a)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_FOC_S_a.png")
        plt.close()
 
        
        return np.array(training_history)

    def export_parameters(self):

        ## Export parameters

        with open(self.params["export_folder"] + '/params.txt', 'a') as the_file:
            for key in self.params.keys():
                if "nn_config" not in key:
                    the_file.write( str(key) + ": " + str(self.params[key]) + '\n')
        nn_config_keys = [x for x in self.params.keys() if "nn_config" in x]

        for nn_config_key in nn_config_keys:
            with open(self.params["export_folder"] + '/params_' + nn_config_key + '.txt', 'a') as the_file:
                for key in self.params[nn_config_key].keys():
                    the_file.write( str(key) + ": " + str(self.params[nn_config_key][key]) + '\n')

    