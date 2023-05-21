import sciris as sc


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
            if isinstance(key, int):
                return self.person(key)
            else:
                errormsg = f'Key "{key}" is not a valid attribute of people'
                raise AttributeError(errormsg)

    def __setitem__(self, key, value):
        ''' Ditto '''
        if self._lock and key not in self.__dict__: # pragma: no cover
            errormsg = f'Key "{key}" is not a current attribute of people, and the people object is locked; see people.unlock()'
            raise AttributeError(errormsg)
        self.__dict__[key] = value
        return
