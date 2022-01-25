(function () {
    "use strict";
    function randId() {
        // Math.random should be unique because of its seeding algorithm.
        // Convert it to base 36 (numbers + letters), and grab the first 9 characters
        // after the decimal.
        return Math.random().toString(36).substr(2, 9);
    };

    function _typeof(obj) {
        if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") {
            _typeof = function (obj) {
                return typeof obj;
            };
        } else {
            _typeof = function (obj) {
                return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
            };
        }
        return _typeof(obj);
    }

    function styleInject(css, ref) {
        if (ref === void 0) ref = {};
        var insertAt = ref.insertAt;
        if (!css || typeof document === "undefined") {
            return;
        }
        var head = document.head || document.getElementsByTagName("head")[0];
        var style = document.createElement("style");
        style.type = "text/css";
        if (insertAt === "top") {
            if (head.firstChild) {
                head.insertBefore(style, head.firstChild);
            } else {
                head.appendChild(style);
            }
        } else {
            head.appendChild(style);
        }
        if (style.styleSheet) {
            style.styleSheet.cssText = css;
        } else {
            style.appendChild(document.createTextNode(css));
        }
    }
    var css = `.imjoy-window{border-style: solid;border-width: 1px;color: #b3b3b3; width: 100%; height:600px;max-width:100%; max-height:200vh;}
.docsify-run-button,.docsify-run-button span,.fullscreen-button,.fullscreen-button span, .docsify-edit-button,.docsify-edit-button span{cursor:pointer;transition:all .25s ease}
.docsify-run-button,.docsify-edit-button,.fullscreen-button{z-index:1;height: 35px;margin-right: 6px;line-height: 10px;overflow:visible;padding:.65em .8em;border:0;border-radius:0;outline:0;font-size:1em;background:#448aff;color:#fff!important;opacity:0.7}
.docsify-run-button span, .fullscreen-button span, .docsify-edit-button span{border-radius:3px;background:inherit;pointer-events:none}
.docsify-run-button:focus,pre:hover .docsify-run-button, .fullscreen-button:focus,pre:hover .fullscreen-button, .docsify-edit-button:focus,pre:hover .docsify-edit-button{opacity:1}
.docsify-close-button{position: absolute;right: 4px;top: 4px;line-height: 10px;height: 40px;z-index:3;cursor:pointer;padding:.65em .8em;border:0;border-radius:0;outline:0;font-size:1em;background:#448aff;color:#fff!important;}
.docsify-fullscreen-button{position: absolute;right: 42px;top: 4px;line-height: 10px;height: 40px;z-index:3;cursor:pointer;padding:.65em .8em;border:0;border-radius:0;outline:0;font-size:1em;background:#448aff;color:#fff!important;}
.docsify-loader {position: absolute;left: 13px;margin-top: 5px;display: inline-block;transform: translate(-50%, -50%);transform: -webkit-translate(-50%, -50%);transform: -moz-translate(-50%, -50%);transform: -ms-translate(-50%, -50%);border: 6px solid #f3f3f3; /* Light grey */border-top: 6px solid #448aff; /* Blue */border-radius: 50%;width: 30px;height: 30px;animation: spin 2s linear infinite;}
@keyframes spin {0% { transform: rotate(0deg); }100% { transform: rotate(360deg); }}
.docsify-status {position: absolute;left: 130px;display: inline-block;font-size:13px;}
.docsify-progressbar-container{display: inline-block;position: absolute; width: 100%;left: 0; height:3px;color:#000!important;background-color:#f1f1f1!important}
.docsify-status .tooltiptext {visibility: hidden; width: 120px;background-color: black;color: #fff;text-align: center;border-radius: 6px;padding: 5px 0;position: absolute;z-index: 1;}
.docsify-status:hover .tooltiptext {visibility: visible!important;}
.show-code-button{text-align: center; color: var(--docsifytabs-tab-highlight-color); cursor: pointer;}
`;
    styleInject(css);

    const i18n = {
        runButtonText: "Run",
        editButtonText: "Edit",
        errorText: "Error",
        successText: "Done"
    };

    async function runCode(mode, config, code) {
        // make a copy of it

        if (config.lang === 'js') config.lang = 'javascript';
        if (config.lang === 'py') config.lang = 'python';
        if (config.lang === 'ijm') config.lang = 'javascript';

        const makePluginSource = (src, config) => {
            if (config.type && !config._parsed) {
                if(config.type === 'macro'){
                    config.passive = false;
                    src = `
                    async function setup(){
                        const source = \`${src}\`;
                        let ij = await api.getWindow("ImageJ.JS-${config.namespace}")
                        if(!ij){
                            ij = await api.createWindow({src:"https://ij.imjoy.io", name:"ImageJ.JS-${config.namespace}"})
                        }
                        await ij.runMacro(source)
                    }
                    api.export({setup});                    
                    `
                    config.type = 'web-worker';
                }
                const cfg = Object.assign({}, config)
                cfg.api_version = cfg.api_version || "0.1.8";
                cfg.name = cfg.name || (config.id && "Plugin-" + config.id) || "Plugin-" + randId();
                cfg.description = cfg.description || "[TODO: describe this plugin with one sentence.]"
                cfg.tags = cfg.tags || []
                cfg.version = cfg.version || "0.1.0"
                cfg.ui = cfg.ui || ""
                cfg.cover = cfg.cover || ""
                cfg.icon = cfg.icon || "extension"
                cfg.inputs = cfg.inputs || null
                cfg.outputs = cfg.outputs || null
                cfg.env = cfg.env || ""
                cfg.permissions = cfg.permissions || []
                cfg.requirements = cfg.requirements || []
                cfg.dependencies = cfg.dependencies || []
                if (config.type === 'window') {
                    cfg.defaults = {}
                }
                if (!config.lang) {
                    if(cfg.type.includes("python")){
                        config.lang = "python"
                    }
                    else if(cfg.type.includes("javascript")){
                        config.lang = "javascript"
                    }
                    else{
                        console.error('"lang" is not specified, please make sure decorate the code block with the name of the language.')
                    }
                }
                if (config.lang !== 'html')
                    src = `<config lang="json">\n${JSON.stringify(cfg, null, 1)}\n</config>\n<script lang="${config.lang}">\n${src}<\/script>`;
                else
                    src = `<config lang="json">\n${JSON.stringify(cfg, null, 1)}\n</config>\n${src}`;
            }
            return src
        }

        const runPluginSource = async (code, initPlugin, windowId, config) => {
            config = Object.assign({}, config);
            // automatically set passive mode if there is no export statement
            if (config.passive === undefined) {
                if (code && !code.includes("api.export(")) {
                    config.passive = true;
                }
            }
            const isHTML = code.trim().startsWith('<')
            if (isHTML || (config.lang === 'html' && !config.type)) {
                const source_config = await window.imjoy.pm.parsePluginCode(code)
                delete source_config.namespace
                for (const k of Object.keys(source_config)) {
                    config[k] = source_config[k]
                }
                config.passive = source_config.passive || config.passive;
                config._parsed = true;
            } else {
                config._parsed = false;
            }
            const src = makePluginSource(code, config);
            const progressElem = document.getElementById('progress_' + config.namespace)
            if (progressElem) progressElem.style.width = `0%`;
            // disable hot reloading for passive plugin
            if (config.passive) {
                config.hot_reloading = false
            }
     
            try {
                if (config.type === 'window') {
                    const wElem = document.getElementById(windowId)
                    if (wElem) wElem.classList.add("imjoy-window");
                    await window.imjoy.pm.imjoy_api.createWindow(initPlugin, {
                        src,
                        namespace: config.namespace,
                        tag: config.tag,
                        window_id: windowId,
                        w: config.w,
                        h: config.h,
                        hot_reloading: config.hot_reloading
                    })
                } else {
                    const plugin = await window.imjoy.pm.imjoy_api.getPlugin(null, {
                        src,
                        namespace: config.namespace,
                        tag: config.tag,
                        hot_reloading: config.hot_reloading
                    })
                    try {
                        if (plugin.run) {
                            await plugin.run({
                                config: {},
                                data: {}
                            });
                        }
                    } catch (e) {
                        this.showMessage(e.toString())
                    }

                }
            } finally {
                if (progressElem) progressElem.style.width = `100%`;

            }
        }
        if (mode === 'edit') {
            const wElem = document.getElementById(config.window_id)
            if (wElem) wElem.classList.add("imjoy-window");
            const cfg = Object.assign({}, config)
            delete cfg.passive
            delete cfg.editor_height
            let editorWindow;
            let pluginInEditor;
            let stopped;
            const api = window.imjoy.pm.imjoy_api;
            cfg.ui_elements = {
                save: {
                    _rintf: true,
                    type: 'button',
                    label: "Save",
                    visible: false,
                    icon: "content-save",
                    callback(content) {
                        console.log(content)
                    }
                },
                run: {
                    _rintf: true,
                    type: 'button',
                    label: "Run",
                    icon: "play",
                    visible: true,
                    shortcut: 'Shift-Enter',
                    async callback(content) {
                        try {
                            editorWindow.setLoader(true);
                            editorWindow.updateUIElement('stop', {
                                visible: true
                            })
                            api.showProgress(editorWindow, 0);
                            // make an exception for imagej macro and try to reuse existing app
                            if(config.type !== 'macro'){
                                const outputContainer = document.getElementById('output_' + config.namespace)
                                outputContainer.innerHTML = "";
                            }
       
                            config.hot_reloading = true;
                            pluginInEditor = await runPluginSource(content, editorWindow, null, config)
                            if (stopped) {
                                pluginInEditor = null;
                                return;
                            }
                            if (pluginInEditor && pluginInEditor.run) {
                                return await pluginInEditor.run({
                                    config: {},
                                    data: {}
                                });
                            }
                            if (stopped) {
                                pluginInEditor = null;
                                return;
                            }
                        } catch (e) {
                            api.showMessage(editorWindow, "Failed to load plugin, error: " + e.toString());
                        } finally {
                            editorWindow.updateUIElement('stop', {
                                visible: false
                            })
                            editorWindow.setLoader(false);
                            api.showProgress(editorWindow, 100);
                        }
                    }
                },
                stop: {
                    _rintf: true,
                    type: 'button',
                    label: "Stop",
                    style: "color: #ff0080cf;",
                    icon: "stop",
                    visible: false,
                    async callback() {
                        stopped = true;
                        await editorWindow.setLoader(false);
                        await editorWindow.updateUIElement('stop', {
                            visible: false
                        })
                    }
                },
                export: {
                    _rintf: true,
                    type: 'button',
                    label: "Export",
                    icon: "file-download-outline",
                    visible: true,
                    async callback(content) {
                        const fileName = (pluginInEditor && pluginInEditor.config.name && pluginInEditor.config.name + '.imjoy.html') || config.name + '.imjoy.html' || "myPlugin.imjoy.html";
                        await api.exportFile(editorWindow, makePluginSource(content, config), fileName);
                    }
                }
            }
            editorWindow = await imjoy.pm.imjoy_api.createWindow(null, {
                src: 'https://if.imjoy.io/',
                config: cfg,
                data: {
                    code,
                },
                window_id: cfg.window_id,
                namespace: cfg.namespace
            })

            if (config.editor_height) document.getElementById(editorWindow.config.window_id).style.height = config.editor_height;
        } else if (mode === 'run') {
            await runPluginSource(code, null, config.window_id, config)
        } else {
            throw "Unsupported mode: " + mode
        }
    }

    function execute(preElm, mode) {
        mode = mode || 'run';
        var codeElm = preElm.querySelector("code");
        const code = codeElm.textContent || codeElm.innerText;
        const showCodeBtn = preElm.querySelector('.show-code-button');

        showCodeBtn.style.display = 'none';

        try {

            const id = randId();
            preElm.pluginConfig = preElm.pluginConfig || {};
            preElm.pluginConfig.id = id;

            preElm.pluginConfig.namespace = id;
            // for some reason, the data-lang attribute disappeared in the newer version
            preElm.pluginConfig.lang = preElm.getAttribute('data-lang');

            const outputFullscreenElm = preElm.querySelector(".fullscreen-button");
            outputFullscreenElm.onclick = () => {
                const outputElem = document.getElementById('output_' + id);
                if (outputElem.requestFullscreen) {
                    outputElem.requestFullscreen();
                } else if (outputElem.webkitRequestFullscreen) {
                    /* Safari */
                    outputElem.webkitRequestFullscreen();
                } else if (outputElem.msRequestFullscreen) {
                    /* IE11 */
                    outputElem.msRequestFullscreen();
                }
            }
            let hideCodeBlock = preElm.pluginConfig.hide_code_block;
            if (mode === 'edit') {
                // remove the github corner in edit mode
                const githubCorner = document.querySelector('.github-corner')
                if (githubCorner) githubCorner.parentNode.removeChild(githubCorner);
            }
            const customElements = preElm.querySelectorAll(":scope > div[id]")
            for (const elm of customElements) {
                preElm.removeChild(elm);
            }

            if (mode === 'edit') {
                const customElements = preElm.querySelectorAll(":scope > button")
                for (const elm of customElements) {
                    elm.style.display = "none";
                }
                preElm.pluginConfig.window_id = 'code_' + id;
                preElm.insertAdjacentHTML('afterBegin', `<div id="${'code_' + id}"></div><div id="${'output_' + id}"></div>`)
                preElm.insertAdjacentHTML('afterBegin', `<button class="docsify-close-button" id="${'close_' + id}">x</button>`);
                preElm.insertAdjacentHTML('afterBegin', `<button class="docsify-fullscreen-button" id="${'fullscreen_' + id}">+</button>`);
                preElm.insertAdjacentHTML('afterBegin', `<div id="${'progress_container_' + id}" style="top: 1px;" class="docsify-progressbar-container"><div class="docsify-progressbar" style="background-color:#2196F3!important;height:3px;" id="${'progress_' + id}"> </div></div>`)
                preElm.insertAdjacentHTML('beforeEnd', `<div class="docsify-status" style="font-size:13px;left: 4px;" id="${'status_' + id}"></div>`);
                const closeElem = document.getElementById('close_' + id)
                const fullscreenElm = document.getElementById('fullscreen_' + id);
                const statusElem = document.getElementById('status_' + id);
                const editorElem = document.getElementById('code_' + id);
                const outputElem = document.getElementById('output_' + id);
                const editorHeight = parseInt(preElm.pluginConfig.editor_height || "600px")
                statusElem.style.top = `${editorHeight - 20}px`;
                editorElem.style.height = `${editorHeight}px`
                editorElem.style.paddingBottom = '10px';
                closeElem.onclick = function () {
                    editorElem.parentNode.removeChild(editorElem)

                    outputElem.parentNode.removeChild(outputElem)
                    if (hideCodeBlock) {
                        showCodeBtn.style.display = "block";
                        codeElm.style.display = "none";
                    } else {
                        showCodeBtn.style.display = "none";
                        codeElm.style.display = "block";
                    }

                    for (const elm of customElements) {
                        elm.style.display = "inline-block";
                    }
                    this.parentNode.removeChild(this)
                    fullscreenElm.parentNode.removeChild(fullscreenElm);
                }
                fullscreenElm.onclick = function () {
                    if (preElm.requestFullscreen) {
                        preElm.requestFullscreen();
                    } else if (preElm.webkitRequestFullscreen) {
                        /* Safari */
                        preElm.webkitRequestFullscreen();
                    } else if (preElm.msRequestFullscreen) {
                        /* IE11 */
                        preElm.msRequestFullscreen();
                    }
                }

                preElm.style.overflow = "hidden";
                outputElem.style.overflow = "auto";

            } else {
                // run mode
                preElm.pluginConfig.window_id = 'output_' + id;
                preElm.insertAdjacentHTML('beforeEnd', `<div id="${'progress_container_' + id}" class="docsify-progressbar-container"><div class="docsify-progressbar" style="background-color:#2196F3!important;height:3px;" id="${'progress_' + id}"> </div></div>`)
                preElm.insertAdjacentHTML('beforeEnd', `<div class="docsify-status" style="margin-top: 8px;" id="${'status_' + id}"/>`);
                preElm.insertAdjacentHTML('beforeEnd', `<div id="${'code_' + id}"></div><div id="${'output_' + id}"></div>`)
                codeElm.style.display = "block";
                showCodeBtn.style.display = 'none';
                const outputElem = document.getElementById('output_' + id);
                outputElem.style.overflow = 'auto';
            }
            const loader = preElm.querySelector(".docsify-loader")
            loader.style.display = "inline-block";
            const runBtn = preElm.querySelector(".docsify-run-button")
            if (runBtn) runBtn.innerHTML = "&nbsp; &nbsp; &nbsp; ";
            if (window.imjoy) {
                runCode(mode, preElm.pluginConfig, code).finally(() => {
                    loader.style.display = "none";
                    const runBtn = preElm.querySelector(".docsify-run-button")
                    if (runBtn) runBtn.innerHTML = i18n.runButtonText;
                    const outputElem = document.getElementById('output_' + id);
                    if (outputElem && outputElem.children.length > 0)
                        outputFullscreenElm.style.display = "inline-block";

                })
            } else {
                window.document.addEventListener("imjoy_app_started", () => {
                    runCode(mode, preElm.pluginConfig, code).finally(() => {
                        loader.style.display = "none";
                        const runBtn = preElm.querySelector(".docsify-run-button")
                        if (runBtn) runBtn.innerHTML = i18n.runButtonText;
                    })
                })
            }

            if (hideCodeBlock || mode === 'edit') {
                codeElm.style.display = "none";
                if (mode !== 'edit') {
                    showCodeBtn.style.display = "block";

                }
            }
            document.addEventListener("fullscreenchange", function (e) {

                const fullScreenMode = document.fullScreen || document.mozFullScreen || document.webkitIsFullScreen;
                if (e.target === preElm) {
                    const closeElem = document.getElementById('close_' + id)
                    const fullscreenElm = document.getElementById('fullscreen_' + id);
                    const editorElem = document.getElementById('code_' + id);
                    const outputElem = document.getElementById('output_' + id);
                    const statusElem = document.getElementById('status_' + id);
                    if (fullScreenMode) {
                        closeElem.style.display = "none";
                        fullscreenElm.style.display = "none";
                        preElm.style.padding = "0";
                        editorElem.style.height = "calc( 100vh - 4px )";
                        editorElem.style.width = "50%";
                        editorElem.style.display = "inline-block";
                        outputElem.style.width = "50%";
                        outputElem._oldHeight = outputElem.style.height;
                        outputElem.style.height = "calc( 100vh - 4px )";
                        outputElem.style.minHeight = "calc( 100vh - 4px )";
                        outputElem.style.display = "inline-block";
                        statusElem.style.top = null
                        statusElem.style.bottom = "1px";
                    } else {
                        closeElem.style.display = "inline-block";
                        fullscreenElm.style.display = "inline-block";
                        preElm.style.padding = "3px";
                        delete outputElem.style.minHeight;
                        editorElem.style.height = preElm.pluginConfig.editor_height || "600px";
                        editorElem.style.width = "100%";
                        editorElem.style.display = "block";
                        outputElem.style.width = "100%";
                        outputElem.style.height = outputElem._oldHeight || '600px';
                        outputElem.style.display = "block";
                        statusElem.style.bottom = null
                        const editorHeight = parseInt(preElm.pluginConfig.editor_height || "600px")
                        statusElem.style.top = `${editorHeight - 20}px`;
                        preElm.scrollIntoView();
                    }
                    return
                }
                const outputElem = document.getElementById('output_' + id);
                if (e.target === outputElem) {
                    if (fullScreenMode) {
                        outputElem.style.width = "100%";
                        outputElem._oldHeight = outputElem.style.height;
                        outputElem.style.height = "calc( 100vh - 4px )";
                        outputElem.style.display = "block";
                        if(outputElem.children.length === 1 && outputElem.children[0])outputElem.children[0].style.height = "100%";
                    } else {
                        outputElem.style.width = "100%";
                        outputElem.style.height = outputElem._oldHeight || '600px';
                        outputElem.style.display = "block";
                        if(outputElem.children.length === 1 && outputElem.children[0]) outputElem.children[0].style.height = null
                        outputElem.scrollIntoView();
                    }
                }
            });
        } catch (err) {
            console.error("docsify-run-code: ".concat(err));
        }
    }

    function initializeRunButtons() {
        var targetElms = Array.apply(null, document.querySelectorAll("pre"));

        var template = ['<button class="docsify-run-button">', '<span class="label">'.concat(i18n.runButtonText, "</span>"), "</button>",
            '<button class="docsify-edit-button">', '<span class="label">'.concat(i18n.editButtonText, "</span>"), "</button>",
            '<div class="docsify-loader"></div>',
            '<button class="fullscreen-button" style="position:absolute; right:0px">+</button>'
        ].join("");

        targetElms.forEach(function (elm) {
            try {
                // patch for rouge syntax highlighter
                if(elm.parentElement && elm.parentElement.parentElement && elm.parentElement.classList.contains('highlight')){
                    elm = elm.parentElement.parentElement
                }

                let tmp = elm.previousSibling && elm.previousSibling.previousSibling;
                if (!tmp || tmp.nodeName !== "#comment" || !tmp.nodeValue.trim().startsWith("ImJoyPlugin")) {
                    // in case there is no empty line
                    // the comment will be nested in the previous sibling
                    if (tmp.childNodes[tmp.childNodes.length - 1].nodeName === "#comment") {
                        tmp = tmp.childNodes[tmp.childNodes.length - 1]
                        if (!tmp.nodeValue.trim().startsWith("ImJoyPlugin")) return;
                    }
                    else {
                        return
                    }
                }
                const ctrlStr = tmp.nodeValue.trim()
                if (ctrlStr === 'ImJoyPlugin') {
                    elm.pluginConfig = {};
                } else {
                    elm.pluginConfig = JSON.parse(ctrlStr.split(':').slice(1).join(":") || "{}");
                }
                elm.insertAdjacentHTML("beforeEnd", template);

                elm.querySelector(".docsify-loader").style.display = "none";
                elm.querySelector(".fullscreen-button").style.display = "none";
                elm.style.position = 'relative';

                const codeElm = elm.querySelector("code");
                codeElm.insertAdjacentHTML('beforeBegin', `<div class="show-code-button">+ show source code</div>`);
                const showCodeBtn = elm.querySelector('.show-code-button');
                showCodeBtn.onclick = () => {
                    codeElm.style.display = 'block';
                    showCodeBtn.style.display = 'none';
                }
                if (elm.pluginConfig.hide_code_block) {

                    codeElm.style.display = 'none';
                } else {
                    showCodeBtn.style.display = 'none'
                }
                if (elm.pluginConfig.startup_mode) {
                    const mode = elm.pluginConfig.startup_mode;
                    execute(elm, mode, true)
                    delete elm.pluginConfig.startup_mode
                }


            } catch (e) {
                console.error(e)
            }

        });

        document.body.addEventListener("click", function (evt) {
            const isRunCodeButton = evt.target.classList.contains("docsify-run-button");
            const isEditCodeButton = evt.target.classList.contains("docsify-edit-button");
            if (isRunCodeButton || isEditCodeButton) {
                var buttonElm = evt.target.tagName === "BUTTON" ? evt.target : evt.target.parentNode;
                const mode = isEditCodeButton ? "edit" : "run";
                execute(buttonElm.parentNode, mode)
            }
        });
    }
    const menuElem = document.createElement('div');
    menuElem.id = "menu-container";
    document.body.appendChild(menuElem);
    const stylePatch = document.createElement('style');
    stylePatch.textContent = `
    .imjoy-dialog-control {
        padding: 0px;
        line-height: 10px;
        color: white!important;
    }`;
    document.head.append(stylePatch);
    loadImJoyBasicApp({
        process_url_query: true,
        show_window_title: false,
        show_progress_bar: true,
        show_empty_window: true,
        menu_style: { position: "absolute", right: "10px", top: "74px" },
        window_style: { width: '100%', height: '100%' },
        main_container: null,
        menu_container: "menu-container",
        window_manager_container: null, // "window-container",
        imjoy_api: {
            async createWindow(_plugin, config) {
                let output;
                if (_plugin && _plugin.config.namespace) {
                    if (_plugin.config.namespace) {
                        const outputContainer = document.getElementById('output_' + _plugin.config.namespace)
                        if (!config.dialog && (!config.window_id || !document.getElementById(config.window_id))) {
                            output = document.createElement('div')
                            output.id = randId();
                            output.classList.add('imjoy-window');
                            outputContainer.style.height = "600px";
                            outputContainer.appendChild(output)
                            config.window_id = output.id
                        }
                    }
                }
                let w;
                // fallback to grid
                if (config.type && config.type.startsWith('imjoy/') || config.type === 'joy') {
                    const grid = await imjoy.pm.createWindow(_plugin, {
                        src: "https://grid.imjoy.io/#/app",
                        window_id: config.window_id,
                        namespace: config.namespace
                    })
                    w = await grid.createWindow(config);
                } else {
                    w = await imjoy.pm.createWindow(_plugin, config)
                }

                return w
            }
        } // override some imjoy API functions here
    }).then(async app => {
        // get the api object from the root plugin
        const api = app.imjoy.api;
        window.imjoy = app.imjoy;
        // if you want to let users to load new plugins, add a menu item
        app.addMenuItem({
            label: "➕ Load Plugin",
            callback() {
                const uri = prompt(
                    `Please type a ImJoy plugin URL`,
                    "https://raw.githubusercontent.com/imjoy-team/imjoy-plugins/master/repository/welcome.imjoy.html"
                );
                if (uri) {
                    app.loadPlugin(uri).then((plugin) => {
                        app.runPlugin(plugin)
                    })
                }
            },
        });
        app.addMenuItem({
            label: "🧩 ImJoy Fiddle",
            async callback() {
                const plugin = await app.loadPlugin("https://if.imjoy.io")
                await app.runPlugin(plugin)
                app.removeMenuItem("🧩 ImJoy Fiddle")
            },
        });

        app.imjoy.pm
            .reloadPluginRecursively({
                uri: "https://imjoy-team.github.io/jupyter-engine-manager/Jupyter-Engine-Manager.imjoy.html"
            })
            .then(enginePlugin => {
                const queryString = window.location.search;
                const urlParams = new URLSearchParams(queryString);
                const engine = urlParams.get("engine");
                const spec = urlParams.get("spec");
                if (engine) {
                    enginePlugin.api
                        .createEngine({
                            name: "MyCustomEngine",
                            nbUrl: engine,
                            url: engine.split("?")[0]
                        })
                        .then(() => {
                            console.log("Jupyter Engine connected!");
                        })
                        .catch(e => {
                            console.error("Failed to connect to Jupyter Engine", e);
                        });
                } else {
                    enginePlugin.api
                        .createEngine({
                            name: "MyBinder Engine",
                            url: "https://mybinder.org",
                            spec: spec || "oeway/imjoy-binder-image/master"
                        })
                        .then(() => {
                            console.log("Binder Engine connected!");
                        })
                        .catch(e => {
                            console.error("Failed to connect to MyBinder Engine", e);
                        });
                }
            });

        initializeRunButtons();
    });

})();
