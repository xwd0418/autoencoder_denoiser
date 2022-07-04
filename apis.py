import os
def get_cuda_num():
    if os.getenv('cuda_in_use') == None:
        os.environ['cuda_in_use'] = '1'
        return '0'
    if os.getenv('cuda_in_use') == "0":
        os.environ['cuda_in_use'] = '1'
        return  '0'
    if os.getenv('cuda_in_use') == "1":
        os.environ['cuda_in_use'] = '2'
        return  '1'
    
def release_cuda():
    if os.getenv('cuda_in_use') == None:
        raise Exception("why no cuda env variable???")
    if os.getenv('cuda_in_use') == "0":
        raise Exception("why no cuda is in use???")
    if os.getenv('cuda_in_use') == "1":
        os.environ['cuda_in_use'] = '0'
        return 
    if os.getenv('cuda_in_use') == "2":
        os.environ['cuda_in_use'] = '1'
        return   
        