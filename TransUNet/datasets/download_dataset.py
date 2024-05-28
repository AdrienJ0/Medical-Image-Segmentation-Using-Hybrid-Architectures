import synapseclient 
import synapseutils 
 
syn = synapseclient.Synapse() 
syn.login('Adrien_J','H9NYS6A}Ch)a;V&') 
averaged_testing_images = synapseutils.syncFromSynapse(syn, 'syn10284975') 
averaged_training_images = synapseutils.syncFromSynapse(syn, 'syn10285054') 
averaged_training_labels = synapseutils.syncFromSynapse(syn, 'syn10285076') 

