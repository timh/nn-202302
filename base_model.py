from typing import Dict, List

from torch import nn

class BaseModel(nn.Module):
    # this should be set by child classes.
    _metadata_fields: List[str]

    """
    (optional): fields to be saved/loaded on with torch.save/load. if not set,
    _metadata_dict will be used.
    """
    _statedict_fields: List[str]

    def metadata_dict(self) -> Dict[str, any]:
        res: Dict[str, any] = dict()
        res['repr'] = repr(self)
        res['class_name'] = type(self).__name__

        for field in self._metadata_fields:
            res[field] = getattr(self, field)
            
        return res
    
    def state_dict(self, *args, **kwargs) -> Dict[str, any]:
        if hasattr(self, '_statedict_fields'):
            res = {field: getattr(self, field) for field in self._statedict_fields}
        else:
            res = self.metadata_dict()
        
        res = res.copy()
        res.update(super().state_dict(*args, **kwargs))
        return res
    
    def load_state_dict(self, state_dict: Dict[str, any], strict: bool = True):
        filtered_state_dict = state_dict.copy()
        
        fields = getattr(self, '_statedict_fields', self._metadata_fields)
        for field in fields:
            if strict:
                val = state_dict.pop(field)
            else:
                val = state_dict.get(field, None)

            setattr(self, field, val)

        return super().load_state_dict(state_dict, strict)
    