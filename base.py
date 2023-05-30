import sciris as sc
import numpy as np
import numba as nb
import operator as op


__all__ = ['ParsObj', 'BaseParams', 'BaseSim', 'BasePop']


class ParsObj:
    '''
    Base class for objects that perform operations with a self.pars dictionary
    '''

    def __init__(self, pars):
        return

    def __getitem__(self, key):
        ''' Allow sim['par_name'] instead of sim.pars['par_name'] '''
        try:
            return self.pars[key]
        except:
            all_keys = '\n'.join(list(self.pars.keys()))
            errormsg = f'Key "{key}" not found; available keys:\n{all_keys}'
            raise sc.KeyNotFoundError(errormsg)

    def __setitem__(self, key, value):
        ''' Ditto '''
        if key in self.pars:
            self.pars[key] = value
        else:
            all_keys = '\n'.join(list(self.pars.keys()))
            errormsg = f'Key "{key}" not found; available keys:\n{all_keys}'
            raise sc.KeyNotFoundError(errormsg)
        return


class BaseParams(ParsObj):
    '''
    Boilerplate methods for parameter objects.
    '''

    def __init__(self):
        super().__init__(self)


class BaseSim(ParsObj):
    '''
    Base simulation object. Handles everything that is not specifically
    ABM related.
    '''

    def __init__(self, default_pars):
        super().__init__(default_pars)


class BasePop:
    '''
    Base population object.
    '''

    def __getitem__(self, key):
        '''
        Allow people['attr'] instead of getattr(people, 'attr')
        If the key is an integer, alias `people.person()` to return a `Person` instance
        '''

        try:
            return self.__dict__[key]
        except: # pragma: no cover
            errormsg = f'Key "{key}" is not a valid attribute of the population'
            raise AttributeError(errormsg)

    def __setitem__(self, key, value):
        ''' Ditto '''
        if self._lock and key not in self.__dict__: # pragma: no cover
            errormsg = f'Key "{key}" is not a current attribute of people, and the people object is locked; see people.unlock()'
            raise AttributeError(errormsg)
        self.__dict__[key] = value
        return

    def true(self, key):
        ''' Return indices matching the condition '''
        return self[key].nonzero()[0]

    def true_and(self, key, key2):
        ''' Return indices that are true for both keys '''
        output = np.logical_and(self[key], self[key2])
        return output.nonzero()[0]

    def false(self, key):
        ''' Return indices not matching the condition '''
        return (~self[key]).nonzero()[0]

    def defined(self, key):
        ''' Return indices of people who are not-nan '''
        return (~np.isnan(self[key])).nonzero()[0]

    def undefined(self, key):
        ''' Return indices of people who are nan '''
        return np.isnan(self[key]).nonzero()[0]

    def defined_str(self, key):
        ''' Return indices of people who are not-nan '''
        return (~np.equal(self[key], 'nan')).nonzero()[0]

    def undefined_str(self, key):
        ''' Return indices of people who are nan '''
        return np.equal(self[key], 'nan').nonzero()[0]

    def count(self, key):
        ''' Count the number of people for a given key '''
        return np.count_nonzero(self[key])

    def itrue(self, arr, inds):
        '''
        Returns the indices that are true in the array -- name is short for indices[true]

        Args:
            arr (array): a Boolean array, used as a filter
            inds (array): any other array (usually, an array of indices) of the same size

        **Example**::

            inds = cv.itrue(np.array([True,False,True,True]), inds=np.array([5,22,47,93]))
        '''
        return inds[arr]

    def ifalse(self, arr, inds):
        '''
        Returns the indices that are true in the array -- name is short for indices[false]

        Args:
            arr (array): a Boolean array, used as a filter
            inds (array): any other array (usually, an array of indices) of the same size

        **Example**::

            inds = cv.ifalse(np.array([True,False,True,True]), inds=np.array([5,22,47,93]))
        '''
        return inds[np.logical_not(arr)]

    def idefined(self, arr, inds):
        '''
        Returns the indices that are defined in the array -- name is short for indices[defined]

        Args:
            arr (array): any array, used as a filter
            inds (array): any other array (usually, an array of indices) of the same size

        **Example**::

            inds = cv.idefined(np.array([3,np.nan,np.nan,4]), inds=np.array([5,22,47,93]))
        '''
        return inds[~np.isnan(arr)]

    def iundefined(self, arr, inds):
        '''
        Returns the indices that are undefined in the array -- name is short for indices[undefined]

        Args:
            arr (array): any array, used as a filter
            inds (array): any other array (usually, an array of indices) of the same size

        **Example**::

            inds = cv.iundefined(np.array([3,np.nan,np.nan,4]), inds=np.array([5,22,47,93]))
        '''
        return inds[np.isnan(arr)]

    def itruei(self, arr, inds):
        '''
        Returns the indices that are true in the array -- name is short for indices[true[indices]]

        Args:
            arr (array): a Boolean array, used as a filter
            inds (array): an array of indices for the original array

        **Example**::

            inds = cv.itruei(np.array([True,False,True,True,False,False,True,False]), inds=np.array([0,1,3,5]))
        '''
        return inds[arr[inds]]

    def ifalsei(self, arr, inds):
        '''
        Returns the indices that are false in the array -- name is short for indices[false[indices]]

        Args:
            arr (array): a Boolean array, used as a filter
            inds (array): an array of indices for the original array

        **Example**::

            inds = cv.ifalsei(np.array([True,False,True,True,False,False,True,False]), inds=np.array([0,1,3,5]))
        '''
        return inds[np.logical_not(arr[inds])]

    def idefinedi(self, arr, inds):
        '''
        Returns the indices that are defined in the array -- name is short for indices[defined[indices]]

        Args:
            arr (array): any array, used as a filter
            inds (array): an array of indices for the original array

        **Example**::

            inds = cv.idefinedi(np.array([4,np.nan,0,np.nan,np.nan,4,7,4,np.nan]), inds=np.array([0,1,3,5]))
        '''
        return inds[~np.isnan(arr[inds])]

    def iundefinedi(self, arr, inds):
        '''
        Returns the indices that are undefined in the array -- name is short for indices[defined[indices]]

        Args:
            arr (array): any array, used as a filter
            inds (array): an array of indices for the original array

        **Example**::

            inds = cv.iundefinedi(np.array([4,np.nan,0,np.nan,np.nan,4,7,4,np.nan]), inds=np.array([0,1,3,5]))
        '''
        return inds[np.isnan(arr[inds])]

    def node_in_cap(self, in_list, nodes):
        ''' Return inds of the first instance of each node in nodes '''
        output = list()
        unique_nodes, num_unique = np.unique(nodes, return_counts=True)
        for i, node in enumerate(unique_nodes):
            curr_inds = np.where(in_list == node)[0]
            for i in curr_inds[0:num_unique[i]]:
                output.append(i)
        return np.array(output, dtype=np.int32)

    def node_ag(self, key, nodes, f):
        ''' Return inds of agents at a given node type '''
        output = list()
        for ind, item in enumerate(self[key]):
            if op.f(nodes[item], 6):
                output.append(ind)
        return np.array(output, dtype=np.int32)

    def count_node(self, nodes):
        ''' Returns both a list of inds that correspond to the agents at the
        nodes given and a dictionary of the nodes and the indices that are at
        those nodes '''
        output = np.zeros(self.pars['pop_size'], dtype=np.int32)
        out_dict = dict()
        for i, node in enumerate(nodes):
            out_dict[node] = np.where(self[node] == 1)[0]
            output += self[node]
        return (np.where(output == 1)[0], out_dict)

    def count_node_if(self, nodes, key2):
        ''' Return inds of agents at a given node type and a dictionary of
        nodes and agents at those nodes '''
        out_nodes, out_dict = self.count_node(nodes)
        output = np.multiply(out_nodes, self[key2][out_nodes])
        out_ind = output.nonzero()[0]
        return (output[out_ind].astype(np.int32), out_dict)

    def find_node(self, agent, nodes):
        ''' Find the node that the agent currently occupies '''
        for node in nodes:
            if self[node][agent] == 1:
                return node
        raise ValueError(f"Agent {agent} not found in given nodes.")
