import io
import torch
import base64
from PIL import Image
from flask_cors import CORS
from threading import Thread
from .environment import Environment
import torchvision.transforms as transforms
from flask import Flask, jsonify, request, send_from_directory


class Server:

    def __init__(self, env: Environment, root: str = 'viewer/www', port: int = 5001):
        self.env = env
        self.env.using_server = True  # forcing
        self.root = root
        self.root_css = root + "/static/css"
        self.root_js = root + "/static/js"
        self.port = port
        self.app = Flask(__name__, template_folder=self.root)
        CORS(self.app)  # to handle cross-origin requests (needed for development)
        self.register_routes()
        self.thumb_transforms = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64)])

        # starting a new thread
        thread = Thread(target=self.__run_server)
        thread.start()

    def __run_server(self):
        self.app.run(host='0.0.0.0', port=self.port, threaded=True, debug=False)  # Run Flask with threading enabled

    def register_routes(self):
        self.app.add_url_rule('/', view_func=self.serve_index, methods=['GET'])
        self.app.add_url_rule('/<path:filename>', view_func=self.serve_root, methods=['GET'])
        self.app.add_url_rule('/static/css/<path:filename>', view_func=self.serve_static_css, methods=['GET'])
        self.app.add_url_rule('/static/js/<path:filename>', view_func=self.serve_static_js, methods=['GET'])
        self.app.add_url_rule('/get_play_pause_status', view_func=self.get_play_pause_status, methods=['GET'])
        self.app.add_url_rule('/ask_to_pause', view_func=self.ask_to_pause, methods=['GET'])
        self.app.add_url_rule('/ask_to_play', view_func=self.ask_to_play, methods=['GET'])
        self.app.add_url_rule('/get_env_name', view_func=self.get_env_name, methods=['GET'])
        self.app.add_url_rule('/get_summary', view_func=self.get_summary, methods=['GET'])
        self.app.add_url_rule('/get_authority', view_func=self.get_authority, methods=['GET'])
        self.app.add_url_rule('/get_behav', view_func=self.get_behav, methods=['GET'])
        self.app.add_url_rule('/get_behav_status', view_func=self.get_behav_status, methods=['GET'])
        self.app.add_url_rule('/get_list_of_agents', view_func=self.get_list_of_agents, methods=['GET'])
        self.app.add_url_rule('/get_list_of_streams', view_func=self.get_list_of_streams, methods=['GET'])
        self.app.add_url_rule('/get_stream', view_func=self.get_stream, methods=['GET'])
        self.app.add_url_rule('/get_console', view_func=self.get_console, methods=['GET'])

    @staticmethod
    def pack_data(_data):
        _type = type(_data).__name__ if _data is not None else "none"

        def is_tensor_or_list_of_tensors(_d):
            if isinstance(_d, list) and len(_d) > 0 and isinstance(_d[0], torch.Tensor):
                return True
            elif isinstance(_d, torch.Tensor):
                return True
            else:
                return False

        def is_pil_or_list_of_pils(_d):
            if isinstance(_d, list) and len(_d) > 0 and isinstance(_d[0], Image.Image):
                return True
            elif isinstance(_d, Image.Image):
                return True
            else:
                return False

        # list of pytorch tensors (or nones)
        def encode_tensor_or_list_of_tensors(__data):
            __type = ""

            if isinstance(__data, list) and len(__data) > 0 and isinstance(__data[0], torch.Tensor):
                found_tensor = False
                __data_b64 = []
                for __tensor in __data:
                    if __tensor is not None:
                        if not found_tensor:
                            found_tensor = True
                            __type = "list_" + type(__data[0]).__name__ + "_" + __data[0].dtype.__str__().split('.')[-1]

                        __data_b64.append(base64.b64encode(__tensor.detach().cpu().numpy().tobytes()).decode('utf-8'))
                    else:
                        __data_b64.append(None)  # there might be some None in some list elements...
                if not found_tensor:
                    __type = "none"
                __data = __data_b64

            # pytorch tensor
            if isinstance(__data, torch.Tensor):
                __type = __data.dtype.__str__().split('.')[-1]
                __data = base64.b64encode(__data.detach().cpu().numpy()).decode('utf-8')

            return __data, __type

        # list of PIL images (or nones)
        def encode_pil_or_list_of_pils(__data):
            __type = ""

            if isinstance(__data, list) and len(__data) > 0 and isinstance(__data[0], Image.Image):
                found_image = False
                _data_b64 = []
                for __img in __data:
                    if __img is not None:
                        if not found_image:
                            found_image = True
                            __type = "list_png"

                        buffer = io.BytesIO()
                        __img.save(buffer, format="PNG", optimize=True, compress_level=9)
                        buffer.seek(0)
                        _data_b64.append(f"data:image/png;base64,{base64.b64encode(buffer.read()).decode('utf-8')}")
                    else:
                        _data_b64.append(None)  # there might be some None in some list elements...
                if not found_image:
                    __type = "none"
                __data = _data_b64

            # pil image
            if isinstance(__data, Image.Image):
                __type = "png"
                __buffer = io.BytesIO()
                __data.save(__buffer, format="PNG", optimize=True, compress_level=9)
                __data = f"data:image/png;base64,{base64.b64encode(__buffer.read()).decode('utf-8')}"

            return __data, __type

        # in the case of a dictionary, we look for values that are (list of) tensors/images and encode them;
        # we augment the key name adding "-type", where "type" is the type of the packed data
        if _type == "dict":
            keys = list(_data.keys())
            for k in keys:
                v = _data[k]
                if is_tensor_or_list_of_tensors(v):
                    v_encoded, v_type = encode_tensor_or_list_of_tensors(v)
                    del _data[k]
                    k = k + "-" + v_type
                    _data[k] = v_encoded
                elif is_pil_or_list_of_pils(v):
                    v_encoded, v_type = encode_pil_or_list_of_pils(v)
                    del _data[k]
                    k = k + "-" + v_type
                    _data[k] = v_encoded
        else:
            if is_tensor_or_list_of_tensors(_data):
                _data, _data_type = encode_tensor_or_list_of_tensors(_data)
                _type += "_" + _data_type
            elif is_pil_or_list_of_pils(_data):
                _data, _data_type = encode_pil_or_list_of_pils(_data)
                _type += "_" + _data_type
            else:
                pass

        # generate JSON for the whole data, where some of them might have been base64 encoded (tensors/images)
        return jsonify({"data": _data, "type": _type})

    def serve_index(self):
        return send_from_directory(self.root, 'index.html')

    def serve_root(self, filename):
        return send_from_directory(self.root, filename)

    def serve_static_js(self, filename):
        return send_from_directory(self.root_js, filename)

    def serve_static_css(self, filename):
        return send_from_directory(self.root_css, filename)

    def get_play_pause_status(self):
        ret = {'status': None,
               'still_to_play': self.env.skip_clear_for}
        if self.env.step == self.env.steps:
            ret['status'] = 'ended'
        elif self.env.step_event.is_set():
            ret['status'] = 'playing'
        elif self.env.wait_event.is_set():
            ret['status'] = 'paused'
        return Server.pack_data(ret)

    def ask_to_play(self):
        steps = int(request.args.get('steps'))
        if steps >= 0:
            self.env.skip_clear_for = steps - 1
        else:
            self.env.skip_clear_for = steps
        self.env.step_event.set()
        return Server.pack_data(self.env.step)

    def ask_to_pause(self):
        self.env.skip_clear_for = 0
        return Server.pack_data(self.env.step)

    def get_env_name(self):
        return Server.pack_data(self.env.name)

    def get_summary(self):
        agent_name = request.args.get('agent_name')
        desc = str(self.env.agents[agent_name]) if agent_name != self.env.name else str(self.env)
        return Server.pack_data(desc)

    def get_authority(self):
        agent_name = request.args.get('agent_name')
        return Server.pack_data(self.env.agents[agent_name].authority)

    def get_behav(self):
        agent_name = request.args.get('agent_name')
        behav = self.env.agents[agent_name].behav if agent_name != self.env.name else self.env.behav
        return Server.pack_data(str(behav.to_graphviz().source))

    def get_behav_status(self):
        agent_name = request.args.get('agent_name')
        behav = self.env.agents[agent_name].behav if agent_name != self.env.name else self.env.behav
        state = behav.states[behav.state][2] if behav.state is not None else None
        action = behav.action[2] if behav.action is not None else None
        return Server.pack_data({'state': state, 'action': action})

    def get_list_of_agents(self):
        agent_name = request.args.get('agent_name')
        agents = self.env.agents[agent_name].known_agents if agent_name != self.env.name else self.env.agents
        return Server.pack_data(list(agents.keys()))

    def get_list_of_streams(self):
        agent_name = request.args.get('agent_name')
        streams = self.env.agents[agent_name].known_streams if agent_name != self.env.name else self.env.streams
        decoupled_streams = []
        for stream in streams.keys():
            decoupled_streams.append(stream + " [y]")
            decoupled_streams.append(stream + " [d]")
        return Server.pack_data(decoupled_streams)

    def get_stream(self):
        agent_name = request.args.get('agent_name')
        stream_name = request.args.get('stream_name')
        since_step = int(request.args.get('since_step'))
        data_id = 0
        if stream_name.endswith(" [y]"):
            stream_name = stream_name[0:stream_name.find(" [y]")]
            data_id = 0
        elif stream_name.endswith(" [d]"):
            stream_name = stream_name[0:stream_name.find(" [d]")]
            data_id = 1
        if agent_name != self.env.name:
            ks, data, last_k, attr = (
                self.env.agents[agent_name].known_streams[stream_name].get_since(since_step, data_id))
        else:
            ks, data, last_k, attr = (
                self.env.streams[stream_name].get_since(since_step, data_id))

        # attr is None if the step index (k) of the stream is -1 (beginning), or if stream is disabled
        if attr is not None:

            # if data has labeled components (and is not "img" and is not "token_ids"),
            # then we take a decision and convert it to a text string
            if (attr.data_type == 'misc' or attr.data_type == 'token_ids') and len(attr) > 0:
                for _i, _data in enumerate(data):
                    data[_i] = attr.data_to_text(_data)

            # if data is of type image, we revert the possibly applied transformation and downscale it
            elif attr.data_type == 'img':
                for _i, _data in enumerate(data):
                    data[_i] = self.thumb_transforms(attr.inv_img_transform(_data.squeeze(0)))

        return Server.pack_data({
            "ks": ks,
            "data": data,
            "last_k": last_k
        })

    def get_console(self):
        agent_name = request.args.get('agent_name')
        agent = self.env.agents[agent_name] if agent_name != self.env.name else self.env
        return Server.pack_data({'output_messages': agent.output_messages,
                                 'output_messages_count': agent.output_messages_count,
                                 'output_messages_last_pos': agent.output_messages_last_pos,
                                 'output_messages_ids': agent.output_messages_ids})
