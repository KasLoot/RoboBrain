"""Official OpenAI payload/auth and shared inference capacity, with mocked transport."""
import asyncio,json,os
from unittest.mock import patch
import httpx
from robobrain.monitor import Backend,load_config
from robobrain.endpoints import model_headers

class Observer:
    references=[];prompt='Classify the image.'
    schema={'type':'object','properties':{'verdict':{'type':'string'}},'required':['verdict'],'additionalProperties':False}
    def current_content(self,frame):return [{'type':'text','text':'Current observation'}]

async def run():
    config=load_config();config.update(base_url='https://api.openai.com/v1',model='gpt-6-luna',reasoning_effort='none',max_in_flight=3)
    active=peak=0;payloads=[]
    async def handle(req):
        nonlocal active,peak
        assert req.headers['Authorization']=='Bearer test-openai-key'
        if req.method=='GET':return httpx.Response(200,json={'data':[{'id':'gpt-6-luna'}]})
        payloads.append(json.loads(req.content));active+=1;peak=max(peak,active)
        try:await asyncio.sleep(.02)
        finally:active-=1
        return httpx.Response(200,json={'choices':[{'message':{'content':'{"verdict":"NEGATIVE"}'}}]})
    with patch.dict(os.environ,{'OPENAI_API_KEY':'test-openai-key'}):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            outcome=Backend(config,client,specialists={});await outcome.check_model()
            progress=Backend(config,client,specialists={},request_gate=outcome.request_gate)
            responses=await asyncio.gather(*(outcome.complete(Observer(),None) for _ in range(20)),progress.complete(Observer(),None))
            assert peak==3
            assert any(r.extensions['model_queue_wait_s']>.02 for r in responses)
            for p in payloads:
                assert p['model']=='gpt-6-luna' and p['reasoning_effort']=='none'
                assert p['response_format']['json_schema']['strict']
                assert 'max_completion_tokens' in p and 'max_tokens' not in p and 'chat_template_kwargs' not in p
            low=Backend({**config,'reasoning_effort':'low'},client,specialists={})
            await low.complete(Observer(),None)
            assert payloads[-1]['reasoning_effort']=='low' and 'temperature' not in payloads[-1]
        assert 'Authorization' not in model_headers('http://localhost:9000/v1')
        assert 'Authorization' not in model_headers('https://api.openai.com.example/v1')
        try:model_headers('http://api.openai.com/v1')
        except ValueError:pass
        else:raise AssertionError('Allowed an insecure API-key destination')
    print('OpenAI monitoring checks passed: gpt-6-luna structured output, host-scoped credentials, compatible parameters, and one shared concurrency budget.')

if __name__=='__main__':asyncio.run(run())
