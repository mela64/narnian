export const DEBUG = false;  // this flag activates console outputs (and might change the endpoint address)

// the console printer :)
export function out(s) {
    if (DEBUG) {
        console.log(s);
    }
}

// convert a Base64-encoded PyTorch tensor into a Javascript typed array
function base64ToTypedArray(base64, type = 'Tensor_float32') {
    const bytes = Uint8Array.from(atob(base64), c => c.charCodeAt(0));
    let typed_array;

    if (type === 'Tensor_float32') {
        typed_array = new Float32Array(bytes.buffer);
    } else if (type === 'Tensor_float64') {
        typed_array = new Float64Array(bytes.buffer);
    } else if (type === 'Tensor_int32') {
        typed_array = new Int32Array(bytes.buffer);
    } else {
        throw new Error(`Unknown tensor type: ${type}`);
    }

    return typed_array;
}

// unpack the data returned by the server, converting it in the format that the rest of the code expects
function unpack_data(received_data) {
    const data = received_data["data"];
    const type = received_data["type"];

    out("[unpack_data] Unpacking received data of type: " + type);
    out("[unpack_data] Received data is: " + JSON.stringify(data, null, 2));

    if (type.startsWith('Tensor')) {  // PyTorch tensor (expected type is "Tensor_float32" for example)
        return base64ToTypedArray(data, type);  // becomes a Javascript typed array

    } else if (type.startsWith("list_Tensor")) {  // list of PyTorch tensors (type is "list_Tensor_float32" for example)
        const tensor_type = type.substring(type.indexOf('_') + 1);
        return data.map(list_elem => base64ToTypedArray(list_elem, tensor_type));  // becomes a list of typed arrays

    } else if (type === "dict") {  // dictionary: each value can also be a PyTorch tensor or a list of PyTorch tensors
        const keys = Object.keys(data);

        for (const key of keys) {
            if (data.hasOwnProperty(key)) {

                // value: PyTorch tensor (expected key is "some_name-Tensor_float32", for example)
                if (key.endsWith("-Tensor_float32") || key.endsWith("-Tensor_float64")
                    || key.endsWith("-Tensor_int32")) {
                    const tensor_type = key.substring(type.lastIndexOf('-') + 1);
                    const new_key = key.substring(0, key.lastIndexOf("-"));
                    data[new_key] = base64ToTypedArray(data, tensor_type);
                    delete data[key];

                // value: list of PyTorch tensors  (expected key is "some_name-list_Tensor_float32", for example)
                } else if (key.endsWith("-list_Tensor_float32") || key.endsWith("-list_Tensor_float64")
                    || key.endsWith("-list_Tensor_int32")) {
                    const tensor_type = key.substring(key.lastIndexOf('-') + 1 + 5);
                    const new_key = key.substring(0, key.lastIndexOf("-"));
                    data[new_key] = data[key].map(list_elem => base64ToTypedArray(list_elem, tensor_type));
                    delete data[key];
                }
            }
        }
        return data;
    } else {
        return data;  // whatever it is
    }
}

// call an API in the rest server, and runs given handlers (if all OK: fcn, if ERROR: fcn_error, FINALLY: fcn_finally)
export function callAPI(api_name, params, fcn, fcn_error, fcn_finally) {
    if (DEBUG) {
        api_name = "http://127.0.0.1:5001" + api_name;
    }

    let queryString = "";
    if (params != null) {
        queryString = new URLSearchParams(params).toString();
    }

    out("[@callAPI] calling " + api_name + " " +
        "with params " + (queryString !== "" ? queryString : '(none)'))

    fetch(`${api_name}${params ? '?' + queryString : ''}`)
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error("Response was not OK!");
            }
        })
        .then((received_data) => {
            fcn(unpack_data(received_data));
        })
        .catch((err) => {
            const err_msg = "Error fetching data by API " + api_name + " " +
                "with params " + (queryString !== "" ? queryString : '(none)') + ". Exception: " + err;
            console.error(err_msg);
            if (fcn_error != null) {
                fcn_error();
            }
            //alert(err_msg);
        })
        .finally(() => {
            if (fcn_finally != null) {
                fcn_finally();
            }
        });
}
