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
tf.random.set_seed(11117)


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
        # All parameters are stored in params which is a dictionary
        self.params  = params 
 
        ## Table 1: Economic Parameters
        self.params["δ"] = 0.02 # subjective discount rate
        self.params["ρ"] = 0.6 # Intertemporal Elasticity of Substitution
        
        self.params["α"] =  0.5 # AI sector production function
        self.params["θ"] =  0.5 # Artificial Labor
        self.params['γ'] =  0.1 # Elasticity of Substitution between Human labor and Artificial Labor
        self.params['ι'] =  0.85 # Proportion of human labor 
        self.params['β'] = 0.3 
        self.params["ψ"] = 1
        
        
        ## Underlying dynamics
        self.params["μ_Z"] =  0.04; self.params["σ_Z"] = 0.035
        self.params["μ_a"] =  0.05 ; self.params["κ_a"] =  6;   self.params["σ_a"] = 0.01
        self.params["μ_g"] =  0.035 ; self.params["κ_g"] = 7;   self.params["σ_g"] = 0.01 
        self.params["ζ"] = 0.02; self.params["ψ_0"] = 0.10  ;self.params["ψ_1"] = 0.5; self.params["σ_κ"] = 0.0078
         
        self.params["A_g"] = 0.12 
        
           
        ## Table 3: State Variable Initial Values and Ranges
        self.params["K_g_0"] = 20.0
        self.params["K_a_0"] = 5.0
        self.params["Z_0"] = 5
        self.params["D_0"] = 5 
        
        self.params["logK_g_min"] = 0.01
        self.params["logK_g_max"] = 4.0
        self.params["logK_a_min"] = 0.01
        self.params["logK_a_max"] = 3.0
        self.params["logZ_min"] = 0.01
        self.params["logZ_max"] = 3.0
        self.params["logD_min"] = 0.01
        self.params["logD_max"] = 2.0 
 
  
 
        ## Create neural networks
        self.v_nn    = FeedForwardSubNet(self.params['v_nn_config'])
        self.i_g_nn  = FeedForwardSubNet(self.params['i_g_nn_config'])
        self.i_a_nn  = FeedForwardSubNet(self.params['i_a_nn_config'])
        self.i_d_nn  = FeedForwardSubNet(self.params['i_d_nn_config'])    

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
    def pde_rhs(self,logK_g, logK_a, logZ,logD,L_g, L_a, w, p):
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
        
        # State variables # will include log xi as a psuede state when consider uncertainty
        state = tf.concat([logK_g, logK_a , logZ,logD], 1) 
        
        ## Evalute neural networks 
        v            = self.v_nn(state)
        i_g          = self.i_g_nn(state)
        i_a          = self.i_a_nn(state)
        i_d          = self.i_d_nn(state)
 
        ## Calculate some variables for proceeding calculation. 

        K_a = tf.reshape(tf.exp(logK_a), [self.params['batch_size'], 1])
        K_g = tf.reshape(tf.exp(logK_g), [self.params['batch_size'], 1])
        Z = tf.reshape(tf.exp(logZ), [self.params['batch_size'], 1])
        D = tf.reshape(tf.exp(logD), [self.params['batch_size'], 1])
 
 
 
 
 
        #########################
        #### Calculate Partial Derivatives
        #########################
        dv_dlogK_a                 = tf.reshape(tf.gradients(v, logK_a, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        dv_ddlogK_a                = tf.reshape(tf.gradients(dv_dlogK_a, logK_a, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        
        dv_dlogK_g                 = tf.reshape(tf.gradients(v, logK_g, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        dv_ddlogK_g                = tf.reshape(tf.gradients(dv_dlogK_g, logK_g, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        
        dv_dlogZ                    = tf.reshape(tf.gradients(v, logZ, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])
        dv_ddlogZ                   = tf.reshape(tf.gradients(dv_dlogZ, logZ, unconnected_gradients='zero')[0], [self.params['batch_size'], 1])


        dv_dlogD                 = tf.reshape(tf.gradients(v, logD, unconnected_gradients='zero')[0], [self.params["batch_size"], 1])
        dv_ddlogD                 = tf.reshape(tf.gradients(dv_dlogD, logD, unconnected_gradients='zero')[0], [self.params["batch_size"], 1])
 
 
        #################
        pv           = self.params["delta"] * v


        p_adpt = 1/(1+tf.exp(-E_t-nu ))


        logC = tf.math.log(A_t) + tf.math.log(0.5) + tf.math.log(  (S_t-N_t ) * p_adpt  + N_t  )

        # Terms for S_t (Technology Frontier)
        v_A_term = Gamma_t* self.params['mu_A']*A_t* dv_dA
        v_AA_term = 0.5* self.params['sigma_A']**2 * A_t**2  * dv_ddA

        # Terms for S_t (Technology Frontier)
        v_S_term =  self.params['mu_S']  * S_t * dv_dS
        v_SS_term = 0.5 * self.params['sigma_S']**2 * S_t**2 * dv_ddS
        # Terms for S_t (Technology Frontier)
        v_E_term =  (  self.params['varphi'] * theta_E- self.params['varphi'] *  tf.math.log(E_t)+0.5*self.params['sigma_E']**2)  * E_t * dv_dE
        v_EE_term = 0.5 * self.params['sigma_E']**2 * E_t**2 * dv_ddE

        # Terms for Gamma_t (Innovation Mitigation)
        v_Gamma_term = self.params['gamma'] * ( self.params['k']* tf.exp(p_adpt*(self.params['theta']  -p_adpt))   - Gamma_t) * dv_dGamma
        v_GammaGamma_term = 0.5 * self.params['sigma_Gamma']**2 * Gamma_t**2 * dv_ddGamma

        # Term for N_t (Aggregate Firm Technology)
        v_N_term = self.params['mu_N'] * N_t * dv_dN
    
        distortion_Gamma = -0.5/xi_Gamma * (   self.params['sigma_Gamma']* Gamma_t* dv_dGamma )**2
        distortion_E = -0.5/xi_E * (   self.params['sigma_E']* E_t* dv_dE )**2
        
        # Put all terms together (Right-hand side of the HJB equation)
        rhs = (
            -pv  # Discounted value
            + logC  # Utility of consumption term
            + v_A_term + v_AA_term  # Aggregate Productivity terms
            + v_S_term + v_SS_term  # Technology Frontier terms
            + v_E_term + v_EE_term  # Technology Frontier terms
            + v_Gamma_term + v_GammaGamma_term  # Innovation Mitigation terms
            + v_N_term  # Aggregate Firm Technology term
            + distortion_Gamma+ distortion_E
        )

        # FOC of control 

        FOC = (S_t-N_t)/( (S_t-N_t) *p_adpt+N_t) + \
            self.params['gamma'] * tf.exp(p_adpt*(self.params['theta']  -p_adpt)) *(   self.params['theta']  -2*p_adpt )* dv_dGamma

        return rhs , FOC

    
     
    @tf.function
    def objective_fn(self,  A_t, S_t, E_t , Gamma_t, N_t,theta_E , xi_E, xi_Gamma, compute_control = False, training = True):

        ## This is the objective function that stochastic gradient descend will try to minimize
        ## It depends on which NN it is training. Controls and value functions have different
        ## objectives.
        rhs , FOC = self.pde_rhs(A_t, S_t, E_t , Gamma_t, N_t,theta_E , xi_E, xi_Gamma, training = True)
 
        if training:    
            ## Take care of nonsensical controls first
            if compute_control:
                return tf.reduce_mean(tf.square(FOC))
                
            else:
                loss = tf.reduce_mean(tf.square(rhs))  + tf.reduce_mean(tf.square(FOC))
                return loss

        else:

            loss = tf.reduce_mean(tf.square(rhs))  + tf.reduce_mean(tf.square(FOC))
            return loss,tf.reduce_mean(tf.square(rhs)),tf.reduce_mean(tf.square(FOC))



    def grad(self, A_t, S_t, E_t , Gamma_t, N_t, theta_E , xi_E, xi_Gamma,compute_control = False, training = True):

        if compute_control:
            with tf.GradientTape(persistent=True) as tape:
                objective = self.objective_fn(A_t, S_t, E_t , Gamma_t, N_t,theta_E , xi_E, xi_Gamma, compute_control, training)
       
            grad = tape.gradient(objective, self.a_nn.trainable_variables  )
            del tape
            return grad
        
        else:
            with tf.GradientTape(persistent=True) as tape:
                objective = self.objective_fn(A_t, S_t, E_t , Gamma_t, N_t, theta_E , xi_E, xi_Gamma,compute_control, training)
            grad = tape.gradient(objective, self.v_nn.trainable_variables)
            del tape

            return grad 

    @tf.function
    def train_step(self):
        A_t, S_t, E_t , Gamma_t, N_t,theta_E , xi_E, xi_Gamma= self.sample()
            

        ## First, train value function
        
        grad = self.grad(A_t, S_t, E_t , Gamma_t, N_t, theta_E , xi_E, xi_Gamma,compute_control= False, training=True)
        self.params["optimizers"][0].apply_gradients(zip(grad, self.v_nn.trainable_variables))

        ## Second, train controls
        grad = self.grad(A_t, S_t, E_t , Gamma_t, N_t, theta_E , xi_E, xi_Gamma,compute_control= True, training=True)

        self.params["optimizers"][1].apply_gradients(zip(grad, self.a_nn.trainable_variables   ))

    def train(self):

        start_time = time.time()
        training_history = []

        # Prepare to store best neural networks and initialize networks
        min_loss = float("inf")
        
        n_inputs = 8
    

        best_v_nn    = FeedForwardSubNet(self.params['value_nn_config'])
        best_v_nn.build( (self.params["batch_size"], n_inputs) ) 
        self.v_nn.build( (self.params["batch_size"], n_inputs) )

        best_a_nn  = FeedForwardSubNet(self.params['nu_nn_config'])
        best_a_nn.build( (self.params["batch_size"], n_inputs) ) 
        self.a_nn.build( (self.params["batch_size"], n_inputs) )


        best_v_nn.set_weights(self.v_nn.get_weights())
        best_a_nn.set_weights(self.a_nn.get_weights()) 
        
        ''' 
        ## Load pretrained weights
        if self.params['pretrained_path'] is not None:
            if "post_tech" in self.params["model_type"] and "post_damage" in self.params["model_type"]:
                print("Loading pretrained model for post-tech post-damage...")
                self.v_nn.load_weights( self.params["pretrained_path"]  + '/v_nn_checkpoint_post_tech_post_damage')
                self.i_g_nn.load_weights( self.params["pretrained_path"]  + '/i_g_nn_checkpoint_post_tech_post_damage')
                self.i_d_nn.load_weights( self.params["pretrained_path"]  + '/i_d_nn_checkpoint_post_tech_post_damage')
                if self.params["n_dims"] == 4:
                    self.i_I_nn.load_weights( self.params["pretrained_path"]  + '/i_I_nn_checkpoint_post_tech_post_damage')

            if "pre_tech" in self.params["model_type"] and "post_damage" in self.params["model_type"]:
                print("Loading pretrained model for pre-tech post-damage...")
                self.v_nn.load_weights( self.params["pretrained_path"]  + '/v_nn_checkpoint_pre_tech_post_damage')
                self.i_g_nn.load_weights( self.params["pretrained_path"]  + '/i_g_nn_checkpoint_pre_tech_post_damage')
                self.i_d_nn.load_weights( self.params["pretrained_path"]  + '/i_d_nn_checkpoint_pre_tech_post_damage')
                if self.params["n_dims"] == 4:
                    self.i_I_nn.load_weights( self.params["pretrained_path"]  + '/i_I_nn_checkpoint_pre_tech_post_damage')

            if "post_tech" in self.params["model_type"] and "pre_damage" in self.params["model_type"]:
                print("Loading pretrained model for post-tech pre-damage...")
                self.v_nn.load_weights( self.params["pretrained_path"]  + '/v_nn_checkpoint_post_tech_pre_damage')
                self.i_g_nn.load_weights( self.params["pretrained_path"]  + '/i_g_nn_checkpoint_post_tech_pre_damage')
                self.i_d_nn.load_weights( self.params["pretrained_path"]  + '/i_d_nn_checkpoint_post_tech_pre_damage')
                if self.params["n_dims"] == 4:
                    self.i_I_nn.load_weights( self.params["pretrained_path"]  + '/i_I_nn_checkpoint_post_tech_pre_damage')

            if "pre_tech" in self.params["model_type"] and "pre_damage" in self.params["model_type"]:
                print("Loading pretrained model for pre-tech pre-damage...")
                self.v_nn.load_weights( self.params["pretrained_path"]  + '/v_nn_checkpoint_pre_tech_pre_damage')
                self.i_g_nn.load_weights( self.params["pretrained_path"]  + '/i_g_nn_checkpoint_pre_tech_pre_damage')
                self.i_d_nn.load_weights( self.params["pretrained_path"]  + '/i_d_nn_checkpoint_pre_tech_pre_damage')
                if self.params["n_dims"] == 4:
                    self.i_I_nn.load_weights( self.params["pretrained_path"]  + '/i_I_nn_checkpoint_pre_tech_pre_damage')
        ''' 

        # begin sgd iteration
        for step in range(self.params["num_iterations"]):
            if step % self.params["logging_frequency"] == 0:
                ## Sample test data
                A_t, S_t, E_t , Gamma_t, N_t ,theta_E , xi_E, xi_Gamma,= self.sample() 
                  
                ## Compute test loss
                test_losses = self.objective_fn(A_t, S_t, E_t , Gamma_t, N_t,theta_E , xi_E, xi_Gamma, training = False)

                ## Update normalization constants
 
                rhs , FOC     = self.pde_rhs(A_t, S_t, E_t , Gamma_t, N_t,theta_E , xi_E, xi_Gamma,)
       
                ## Store best neural networks

                if (test_losses[0] < min_loss):
                    min_loss = test_losses[0]

                    best_v_nn.set_weights(self.v_nn.get_weights())
                    best_a_nn.set_weights(self.a_nn.get_weights())

                ## Generate checkpoints for tensorboard
                if self.params['tensorboard']:
                    grad_v_nn     = self.grad(A_t, S_t, E_t , Gamma_t, N_t,theta_E , xi_E, xi_Gamma, compute_control= False, training=True)
                    grad_controls = self.grad(A_t, S_t, E_t , Gamma_t, N_t,theta_E , xi_E, xi_Gamma, compute_control= True, training=True)

                    with self.test_writer.as_default():

                        ## Export learning rates
                        for optimizer_idx in range(len(self.params['optimizers'])):
                            if "sgd" in self.params['learning_rate_schedule_type']:
                                tf.summary.scalar('learning_rate_' + str(optimizer_idx), self.params["optimizers"][optimizer_idx]._decayed_lr(tf.float32), step = step)
                            else:
                                tf.summary.scalar('learning_rate_' + str(optimizer_idx), self.params["optimizers"][optimizer_idx].lr, step = step)

                        ## Export losses
                        tf.summary.scalar('loss_v', test_losses[0], step = step)
                        tf.summary.scalar('loss_rhs', test_losses[1], step = step)
                        tf.summary.scalar('loss_control', test_losses[2], step = step)
                        ## Export weights and gradients

                        for layer in self.v_nn.layers:
                                                        
                            for W in layer.weights:
                                tf.summary.histogram(W.name + '_weights', W, step = step)

                        for g in range(len(self.v_nn.trainable_variables)):
                            tf.summary.histogram( self.v_nn.trainable_variables[g].name + '_grads', grad_v_nn[g], step = step )

                        for layer in self.a_nn.layers:
                            for W in layer.weights:
                                tf.summary.histogram(W.name + '_weights', W, step = step)

                        for g in range(len(self.a_nn.trainable_variables)):
                            tf.summary.histogram( self.a_nn.trainable_variables[g].name + '_grads', grad_controls[g], step = step )
 
                elapsed_time = time.time() - start_time

                ## Appendinging to training history
                entry = [step] + list(test_losses) 

                training_history.append(entry)

                ## Save training history
                header = 'step,loss_v,loss_rhs,loss_control'

                np.savetxt(  self.params["export_folder"] + '/training_history.csv',
                    training_history,
                    #fmt= ['%d'] + ['%.5e'] * len(test_losses) + ['%.5e','%.5e','%d'], # ['%d', '%.5e', '%.5e', '%.5e', '%d'],
                    delimiter=",",
                    header=header,
                    comments='')
        
            self.train_step()

        ## Use best neural networks 
        self.v_nn.set_weights(best_v_nn.get_weights())
        self.a_nn.set_weights(best_a_nn.get_weights()) 

        ## Export last check point
        self.v_nn.save_weights( self.params["export_folder"] + '/v_nn_checkpoint'  )
        self.a_nn.save_weights( self.params["export_folder"] + '/a_nn_checkpoint'  ) 

        ## Save training history
        header = 'step,loss_v,loss_rhs,loss_control'

        np.savetxt(  self.params["export_folder"] + '/training_history.csv',
            training_history,
            #fmt= ['%d'] + ['%.5e'] * len(test_losses) + ['%.5e','%.5e','%d'], # ['%d', '%.5e', '%.5e', '%.5e', '%d'],
            delimiter=",",
            header=header,
            comments='')

        ## Plot losses

        loss_v_history                   = [history_record[1] for history_record in training_history]
        loss_rhs   = [history_record[2] for history_record in training_history]
        loss_control             = [history_record[3] for history_record in training_history]


        plt.figure()
        plt.title("Test loss: value function")
        plt.plot(loss_v_history)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_v_history.png")
        plt.close()

        plt.figure()
        plt.title("Test loss: controls")
        plt.plot(loss_rhs)
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_rhs.png")
        plt.close()

        plt.figure()
        plt.title("Test loss: dvdY")
        plt.plot(loss_control)
        plt.xscale('log')
        plt.savefig( self.params["export_folder"] + "/loss_controls.png")
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
