from typing import Dict, List

from torch import nn

class BaseModel(nn.Module):
    # this should be set by child classes.
    _metadata_fields: List[str]

    """
    (optional): fields to be saved/loaded on with torch.save/load. if not set,
    _metadata_dict will be used. These should be only the fields that can be used
    to instantiate the model.
    """
    _model_fields: List[str]

    def metadata_dict(self) -> Dict[str, any]:
        res: Dict[str, any] = dict()
        # res['repr'] = repr(self)
        # res['class_name'] = type(self).__name__

        for field in self._metadata_fields:
            res[field] = getattr(self, field)
            
        return res
    
    def model_dict(self, *args, **kwargs) -> Dict[str, any]:
        if not hasattr(self, '_model_fields'):
            print(f"warning: model doesn't have _model_fields")
            return {}

        res = {field: getattr(self, field) for field in self._model_fields}
        
        res = res.copy()
        res.update(super().state_dict(*args, **kwargs))
        return res
    
    def load_model_dict(self, model_dict: Dict[str, any], strict: bool = True):
        model_dict = model_dict.copy()
        
        fields = getattr(self, '_model_fields', self._metadata_fields)
        for field in fields:
            if strict:
                # NOTE: pop with default=None allows loading checkpoints after
                # a new (default valued) constructor argument has been added.
                val = model_dict.pop(field, None)
            else:
                val = model_dict.get(field, None)

            setattr(self, field, val)

        return super().load_state_dict(model_dict, strict)
    