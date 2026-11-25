// tools_catalog.js — 第 19–21 篇共用的「工具目录」：约 45 个应用 / MCP 服务器、两千多个工具（JSON Schema 齐全），
// 以及从同一份描述生成的「用户请求」（训练检索模型 + 评测用，带标注）。
// 工具形态仿照公开的 MCP 服务器与各软件的脚本 API（GitHub / filesystem / git / Playwright / Slack 的官方或参考服务器、
// Blender MCP、Unreal MCP 等）；名字和参数是按真实 API 的结构整理的示意，不保证与官方逐一相同。
// 浏览器与 Node 通用：const TKCAT = …；Node 里 module.exports = TKCAT。
const TKCAT = (() => {
    // ═════════════════════════ 1. 动词：同义说法（中 / 英）、英文描述模板、默认参数 ═════════════════════════
    // 参数小语言：'name:str path:str? format:fbx|gltf|obj? count:int location:vec3 props:obj tags:str[] on:bool'；? = 可选
    const V = {
        list: ['列出,有哪些,列一下,看看所有,显示全部,都有什么,查一下所有', 'list,show all,get all,enumerate', 'List {os} in {app}. Supports an optional name filter and paging.', 'filter:str? limit:int? cursor:str?', 'r'],
        get: ['查看,获取,读取,看一下,查询,看看', 'get,show,read,inspect,describe', 'Get the details and properties of a {o}.', 'id:str', 'r'],
        create: ['新建,创建,添加,加一个,新增,建一个', 'create,add,new,make', 'Create a new {o}.', 'name:str', 'w'],
        delete: ['删除,删掉,移除,去掉', 'delete,remove,drop', 'Delete a {o}. This cannot be undone.', 'id:str', 'd'],
        update: ['修改,更新,改成,设置,调整', 'update,set,change,modify,edit', 'Update one or more properties of a {o}.', 'id:str properties:obj', 'w'],
        rename: ['重命名,改名,改名字', 'rename', 'Rename a {o}.', 'id:str new_name:str', 'w'],
        duplicate: ['复制,拷贝一份,再复制一个', 'duplicate,copy,clone', 'Duplicate a {o}.', 'id:str count:int?', 'w'],
        move: ['移动,挪到,移到', 'move,relocate', 'Move a {o} to another location or folder.', 'id:str destination:str', 'w'],
        import: ['导入,载入,加载,拖进来', 'import,load,bring in', 'Import a {o} from a file on disk.', 'path:str', 'w'],
        export: ['导出,输出,另存为', 'export,save as,output', 'Export a {o} to a file.', 'id:str path:str format:str?', 'w'],
        search: ['搜索,查找,找一下,搜一下', 'search,find,look for,look up', 'Search {os} by keyword. Returns the best matches.', 'query:str limit:int?', 'r'],
        run: ['运行,执行,跑一下,启动', 'run,execute,launch', 'Run a {o}.', 'id:str args:obj?', 'w'],
        build: ['编译,构建,打包', 'build,compile', 'Build or compile the {o}.', 'target:str?', 'w'],
        open: ['打开', 'open', 'Open a {o}.', 'path:str', 'r'],
        save: ['保存,存一下', 'save', 'Save the current {o}.', 'path:str?', 'w'],
        send: ['发送,发,发给,通知,告诉,群发', 'send,post,notify,tell', 'Send a {o}.', 'to:str content:str', 'o'],
        reply: ['回复,回一下', 'reply,respond to', 'Reply to a {o}.', 'id:str content:str', 'o'],
        assign: ['分配给,指派给,交给', 'assign', 'Assign a {o} to a user.', 'id:str assignee:str', 'w'],
        comment: ['评论,留言,加评论', 'comment on,add a comment to', 'Add a comment to a {o}.', 'id:str body:str', 'o'],
        share: ['分享,共享', 'share', 'Share a {o} with users.', 'id:str users:str[] permission:view|edit?', 'o'],
        download: ['下载', 'download', 'Download a {o} to a local path.', 'id:str path:str?', 'r'],
        upload: ['上传', 'upload', 'Upload a local file as a {o}.', 'path:str destination:str', 'o'],
        start: ['启动,开启', 'start', 'Start a {o}.', 'id:str', 'w'],
        stop: ['停止,停掉,关掉', 'stop', 'Stop a running {o}.', 'id:str', 'w'],
        restart: ['重启', 'restart', 'Restart a {o}.', 'id:str', 'w'],
        logs: ['看日志,查日志,日志', 'logs,tail logs of,show logs for', 'Fetch recent log lines of a {o}.', 'id:str lines:int?', 'r'],
        approve: ['批准,同意,审批通过', 'approve', 'Approve a pending {o}.', 'id:str comment:str?', 'o'],
        reject: ['拒绝,驳回', 'reject,decline', 'Reject a pending {o}.', 'id:str reason:str?', 'o'],
        merge: ['合并', 'merge', 'Merge a {o}.', 'id:str method:merge|squash|rebase?', 'd'],
        render: ['渲染,出图', 'render', 'Render the {o} to an image or video file.', 'output:str', 'w'],
        bake: ['烘焙', 'bake', 'Bake the {o}.', 'quality:preview|medium|high|production?', 'w'],
        play: ['播放,预览,试玩', 'play,preview', 'Start playing / previewing the {o}.', '', 'w'],
        select: ['选中,选择', 'select', 'Select {os} in the editor.', 'ids:str[]', 'r'],
        attach: ['挂到,附加到,绑到', 'attach,parent', 'Attach a {o} to a parent.', 'id:str parent:str', 'w'],
        convert: ['转换,转成,转码', 'convert,transcode', 'Convert a {o} to another format.', 'input:str output:str', 'w'],
        resize: ['缩放,改尺寸,调整大小,改分辨率', 'resize,scale', 'Resize a {o}.', 'input:str width:int height:int?', 'w'],
        schedule: ['安排,预约,约一个', 'schedule,book', 'Schedule a {o}.', 'title:str start:str end:str attendees:str[]?', 'o'],
        query: ['查询,统计,查一下', 'query,count,report', 'Query {os} with filters and aggregation.', 'start_date:str end_date:str group_by:str?', 'r'],
        validate: ['检查,校验,验证', 'validate,check,lint', 'Validate a {o} and report problems.', 'id:str', 'r'],
        sync: ['同步,拉取最新', 'sync,pull,update to latest', 'Sync the {o} to the latest revision.', 'path:str?', 'w'],
        submit: ['提交', 'submit,commit', 'Submit the {o}.', 'description:str', 'o'],
        revert: ['还原,撤销修改', 'revert,undo changes to', 'Revert local changes of a {o}.', 'id:str', 'd'],
        lock: ['加锁,锁定', 'lock', 'Lock a {o} so others cannot edit it.', 'id:str', 'w'],
        invite: ['邀请,拉进来', 'invite,add', 'Invite users to a {o}.', 'id:str users:str[]', 'o'],
        find: ['找到,定位,查找', 'find,locate', 'Find {os} by name, class or tag.', 'name:str? class:str? tag:str?', 'r'],
        compile: ['编译', 'compile', 'Compile a {o} and report errors.', 'id:str', 'w'],
        set: ['设置,设成,改成', 'set', 'Set the value of a {o}.', 'name:str value:any', 'w'],
        remove: ['移除,删掉,卸载', 'remove,uninstall', 'Remove a {o}.', 'id:str', 'd'],
        add: ['添加,加一个,新增', 'add', 'Add a {o}.', 'name:str', 'w'],
        read: ['读取,看看内容', 'read', 'Read the contents of a {o}.', 'id:str', 'r'],
        set_transform: ['位置,旋转,缩放,摆到,挪一下', 'transform,location,rotation,scale,move', 'Set location, rotation and scale of an {o}.', 'id:str location:vec3? rotation:vec3? scale:vec3?', 'w'],
        set_parent: ['设父节点,挂到下面,父子关系', 'set parent,parent under', 'Re-parent a {o}.', 'id:str parent:str', 'w'],
        reference: ['引用,参考进来', 'reference', 'Reference a {o} file into the current scene.', 'path:str namespace:str?', 'w'],
        close: ['关闭', 'close', 'Close the {o}.', 'save:bool?', 'w'],
        crop: ['裁剪,裁切', 'crop', 'Crop a {o}.', 'input:str output:str geometry:str', 'w'],
        insert: ['插入,加入', 'insert', 'Insert a {o}.', 'position:str content:any', 'w'],
        apply: ['应用,套用', 'apply', 'Apply a {o}.', 'name:str', 'w'],
        cancel: ['取消', 'cancel', 'Cancel a {o}.', 'id:str reason:str?', 'd'],
        refund: ['退款', 'refund', 'Refund a {o}.', 'id:str amount:num?', 'd'],
        fulfill: ['发货', 'fulfill,ship', 'Mark a {o} as fulfilled / shipped.', 'id:str tracking_number:str?', 'o'],
        archive: ['归档', 'archive', 'Archive a {o}.', 'id:str', 'w'],
        complete: ['完成,标记完成,打勾', 'complete,mark done', 'Mark a {o} as complete.', 'id:str', 'w'],
        pause: ['暂停', 'pause', 'Pause a {o}.', 'id:str', 'w'],
        resume: ['恢复,继续', 'resume', 'Resume a paused {o}.', 'id:str', 'w'],
        publish: ['发布', 'publish', 'Publish a {o}.', 'id:str', 'o'],
        adjust: ['调整,修改数量', 'adjust', 'Adjust a {o}.', 'id:str delta:int', 'w'],
        tag: ['打标签,加标签', 'tag,label', 'Add tags to a {o}.', 'id:str tags:str[]', 'w'],
        void: ['作废', 'void', 'Void a {o}.', 'id:str', 'd'],
    };

    // ═════════════════════════ 2. 应用 / MCP 服务器 ═════════════════════════
    // A(ns, 显示名, 领域, 中文叫法, 英文叫法, 接入方式, 一句话描述, { 对象: [中文叫法, 英文叫法(第 1 个是英文名，第 2 个是复数), '动词 动词 …', 参数覆盖?] }, 自定义动词?, 精确命名的工具?)
    // 自定义动词：{ key: ['中文说法', '英文说法', '英文描述', '参数', r|w|d|o] }（key 的前缀与对象组成工具名，例如 set_intensity + light → set_light_intensity）
    const APPS = [];
    const A = (ns, name, domain, zh, en, via, about, objects, verbs, exact) => APPS.push({ ns, name, domain, zh: zh.split(','), en: en.split(','), via, about, objects, verbs: verbs || {}, exact: exact || [] });

    // —— 游戏引擎与游戏开发 ——
    A('unreal', 'Unreal Engine', 'game', 'UE,UE5,虚幻,虚幻引擎,虚幻编辑器', 'unreal,unreal engine,ue5,ue', '编辑器 Python API / Remote Control API，经 MCP 服务器（编辑器插件）暴露',
        'Control the Unreal Editor: levels, actors, lights, materials, blueprints, assets, sequencer and builds.', {
            level: ['关卡,地图', 'level,levels', 'open save create list', { create: 'name:str template:empty|basic|open_world?' }],
            actor: ['Actor,物体,对象,场景里的东西', 'actor,actors', 'list spawn delete duplicate rename select attach set_transform set_property find', { spawn: 'class:str name:str? location:vec3 rotation:vec3? scale:vec3?', list: 'class_filter:str? name_filter:str? limit:int? cursor:str?' }],
            light: ['灯,灯光,光源,点光源,聚光灯,平行光', 'light,lights', 'spawn list set_intensity set_color set_attenuation toggle_shadows', { spawn: 'type:point|spot|directional|rect name:str? location:vec3 intensity:num?' }],
            static_mesh: ['静态网格,模型,网格体', 'static mesh,static meshes', 'spawn list set_material set_lod'],
            material: ['材质,材质球', 'material,materials', 'create list assign create_instance set_parameter', { assign: 'actor:str material:str slot:int?' }],
            blueprint: ['蓝图,BP', 'blueprint,blueprints', 'create compile open list add_component add_variable', { create: 'name:str parent_class:Actor|Pawn|Character|ActorComponent path:str?' }],
            asset: ['资源,资产,素材', 'asset,assets', 'search import rename move delete fix_redirectors', { import: 'path:str destination:str? type:fbx|texture|audio|auto?' }],
            camera: ['摄像机,相机,镜头', 'camera,cameras', 'spawn set_fov focus'],
            niagara: ['粒子,特效,Niagara', 'niagara system,niagara systems', 'spawn set_parameter list'],
            sequence: ['Sequencer,过场动画,关卡序列', 'level sequence,level sequences', 'create add_track render'],
            widget: ['UMG,UI 控件,界面', 'widget blueprint,widget blueprints', 'create add_element'],
            project: ['工程,项目,包', 'project,projects', 'build_lighting cook package run_tests'],
            pie: ['PIE,运行游戏,试玩', 'play in editor session,PIE sessions', 'play stop'],
            cvar: ['控制台变量,CVar,控制台命令', 'console variable,console variables', 'set get exec'],
            navmesh: ['导航网格,寻路网格', 'navigation mesh,navigation meshes', 'build'],
            viewport: ['视口,编辑器视口', 'viewport,viewports', 'screenshot set_camera'],
        }, {
            set_intensity: ['调亮度,改强度,调强度,调暗,调亮,亮一点,暗一点,强度', 'set intensity,set brightness,dim,brighten,intensity', 'Set the intensity of a {o} (candela for point/spot lights, lux for directional lights).', 'actor:str intensity:num unit:candela|lux|unitless?', 'w'],
            set_color: ['改颜色,调颜色,换颜色,变暖,变冷,色温', 'set color,change color,tint,color temperature', 'Set the light color of a {o}, as RGB or color temperature in Kelvin.', 'actor:str color:vec3? temperature:num?', 'w'],
            set_attenuation: ['衰减半径,照射范围,影响范围', 'attenuation radius,falloff,light range', 'Set the attenuation radius of a {o}.', 'actor:str radius:num', 'w'],
            toggle_shadows: ['阴影,投影,开关阴影', 'shadows,cast shadows,toggle shadows', 'Enable or disable shadow casting of a {o}.', 'actor:str cast_shadows:bool', 'w'],
            set_transform: ['位置,摆到,旋转,缩放,挪一下', 'transform,location,rotation,position,place', 'Set location, rotation and scale of a {o}.', 'actor:str location:vec3? rotation:vec3? scale:vec3?', 'w'],
            set_property: ['属性,改属性,设置属性', 'property,set property', 'Set a property on a {o} by name (any editable UPROPERTY).', 'actor:str property:str value:any', 'w'],
            spawn: ['生成,放一个,摆一个,创建,添加,新建', 'spawn,place,add,create', 'Spawn a {o} in the current level.', 'class:str name:str? location:vec3', 'w'],
            set_material: ['换材质,设置材质', 'set material,change material', 'Set the material of a {o}.', 'actor:str material:str slot:int?', 'w'],
            set_lod: ['LOD,细节层级,减面', 'lod,level of detail', 'Configure LOD settings of a {o}.', 'mesh:str lod_count:int reduction:num?', 'w'],
            create_instance: ['材质实例,MI', 'material instance,instance', 'Create a material instance from a parent {o}.', 'parent:str name:str', 'w'],
            set_parameter: ['参数,改参数', 'parameter,set parameter', 'Set a scalar / vector / texture parameter on a {o}.', 'id:str parameter:str value:any', 'w'],
            add_component: ['加组件,添加组件', 'add component', 'Add a component to a {o}.', 'blueprint:str component_class:str name:str?', 'w'],
            add_variable: ['加变量,添加变量', 'add variable', 'Add a member variable to a {o}.', 'blueprint:str name:str type:bool|int|float|string|vector|object', 'w'],
            fix_redirectors: ['修复重定向器,清理重定向', 'fix up redirectors,redirectors', 'Fix up redirectors left behind after moving {os}.', 'path:str', 'w'],
            set_fov: ['视野,FOV,焦距', 'field of view,fov,focal length', 'Set the field of view of a {o}.', 'camera:str fov:num', 'w'],
            focus: ['对准,聚焦', 'focus,look at', 'Point a {o} at a target actor.', 'camera:str target:str', 'w'],
            add_track: ['加轨道,添加轨道,K 帧', 'add track,keyframe', 'Add an actor or property track to a {o}.', 'sequence:str actor:str property:str?', 'w'],
            add_element: ['加按钮,加文本,添加控件', 'add button,add text,add element', 'Add a child widget (button, text, image…) to a {o}.', 'widget:str type:Button|TextBlock|Image|ProgressBar name:str', 'w'],
            build_lighting: ['构建光照,烘焙光照,打光照', 'build lighting,bake lighting', 'Build static lighting for the current level.', 'quality:preview|medium|high|production?', 'w'],
            cook: ['烘焙内容,Cook,cook 资源', 'cook,cook content', 'Cook content for a target platform.', 'platform:Windows|Android|IOS|PS5|XSX', 'w'],
            package: ['打包,出包', 'package,package project', 'Package the {o} for a platform.', 'platform:Windows|Android|IOS|PS5|XSX configuration:Development|Shipping?', 'w'],
            run_tests: ['跑自动化测试,自动化测试', 'run automation tests,automation tests', 'Run automation tests of the {o}.', 'filter:str?', 'w'],
            exec: ['执行控制台命令,控制台', 'console command,exec', 'Execute a console command in the editor.', 'command:str', 'w'],
            set_camera: ['视角,移动视口', 'viewport camera,move viewport', 'Move the editor {o} camera.', 'location:vec3 rotation:vec3', 'r'],
            screenshot: ['截图,截屏', 'screenshot,capture', 'Capture a screenshot of the {o}.', 'resolution:str?', 'r'],
        }, [
            ['execute_python', '执行 Python,运行 Python 脚本,写个脚本', 'run python,execute python,python script', 'Execute arbitrary Python in the Unreal Editor (unreal module). Escape hatch for anything without a dedicated tool.', 'code:str', 'w'],
            ['get_level_actors', '关卡里所有物体,场景里有什么', 'actors in level,all actors', 'Return actors in the current level with class, name and transform. Paged.', 'class_filter:str? limit:int? cursor:str?', 'r'],
        ]);
    A('unity', 'Unity', 'game', 'Unity,团结引擎,U3D', 'unity,unity editor', 'C# 编辑器脚本（Editor API），经 MCP 服务器暴露',
        'Control the Unity Editor: scenes, GameObjects, components, prefabs, scripts, assets, tests and builds.', {
            scene: ['场景', 'scene,scenes', 'open save create list'],
            gameobject: ['GameObject,物体,对象,游戏对象', 'gameobject,gameobjects', 'create delete find duplicate rename set_transform set_active set_parent add_component'],
            component: ['组件', 'component,components', 'get update remove'],
            prefab: ['预制体,Prefab', 'prefab,prefabs', 'create instantiate apply_overrides unpack'],
            material: ['材质', 'material,materials', 'create assign set_shader set_property'],
            script: ['脚本,C# 脚本,代码', 'C# script,C# scripts', 'create attach read list', { create: 'name:str template:MonoBehaviour|ScriptableObject|Editor path:str?' }],
            asset: ['资源,素材', 'asset,assets', 'search import move delete refresh'],
            build: ['构建,打包,出包', 'player build,player builds', 'build switch_platform'],
            test: ['单元测试,测试', 'test,tests', 'run_editmode run_playmode'],
            playmode: ['Play 模式,运行游戏', 'play mode,play mode', 'enter exit'],
            console: ['控制台,日志', 'console log,console logs', 'read_logs clear'],
            animator: ['动画状态机,Animator', 'animator controller,animator controllers', 'create add_state add_transition'],
            lighting: ['光照,光照贴图', 'lightmap,lightmaps', 'bake'],
            package: ['包,Package', 'package,packages', 'add remove list'],
        }, {
            set_active: ['显示,隐藏,启用,禁用', 'enable,disable,show,hide,set active', 'Enable or disable a {o}.', 'path:str active:bool', 'w'],
            set_parent: ['设父节点,挂到下面', 'set parent,parent under', 'Re-parent a {o}.', 'path:str parent:str', 'w'],
            set_transform: ['位置,旋转,缩放,摆到', 'transform,position,rotation,scale', 'Set position, rotation and scale of a {o}.', 'path:str position:vec3? rotation:vec3? scale:vec3?', 'w'],
            add_component: ['加组件,添加组件,挂组件,加刚体,加碰撞体,加音源', 'add component,add rigidbody,add collider', 'Add a component (e.g. Rigidbody, BoxCollider, AudioSource) to a {o}.', 'path:str type:str', 'w'],
            instantiate: ['实例化,放到场景里', 'instantiate,place', 'Instantiate a {o} in the open scene.', 'prefab:str position:vec3?', 'w'],
            apply_overrides: ['应用修改,Apply', 'apply overrides', 'Apply instance overrides back to the {o} asset.', 'path:str', 'w'],
            unpack: ['解包,Unpack', 'unpack', 'Unpack a {o} instance.', 'path:str completely:bool?', 'w'],
            set_shader: ['换 Shader,设置着色器', 'set shader', 'Set the shader of a {o}.', 'material:str shader:str', 'w'],
            set_property: ['改属性,设置属性', 'set property', 'Set a property of a {o}.', 'material:str property:str value:any', 'w'],
            attach: ['挂到物体上,挂脚本', 'attach script', 'Attach a {o} to a GameObject.', 'script:str path:str', 'w'],
            refresh: ['刷新资源,刷新 AssetDatabase', 'refresh asset database', 'Refresh the AssetDatabase after external changes.', '', 'w'],
            switch_platform: ['切换平台', 'switch platform', 'Switch the active build target.', 'platform:StandaloneWindows64|Android|iOS|WebGL', 'w'],
            run_editmode: ['跑 EditMode 测试,编辑模式测试', 'run edit mode tests', 'Run EditMode {os} with the Unity Test Framework.', 'filter:str?', 'w'],
            run_playmode: ['跑 PlayMode 测试,运行模式测试', 'run play mode tests', 'Run PlayMode {os}.', 'filter:str?', 'w'],
            enter: ['进入 Play,开始运行', 'enter play mode', 'Enter {o}.', '', 'w'],
            exit: ['退出 Play,停止运行', 'exit play mode', 'Exit {o}.', '', 'w'],
            read_logs: ['读日志,看报错', 'read console,read errors', 'Read recent {os} (errors, warnings).', 'type:error|warning|log|all? limit:int?', 'r'],
            clear: ['清空日志', 'clear console', 'Clear the {o}.', '', 'w'],
            add_state: ['加状态', 'add state', 'Add a state to an {o}.', 'controller:str name:str clip:str?', 'w'],
            add_transition: ['加过渡,连线', 'add transition', 'Add a transition between two states of an {o}.', 'controller:str from:str to:str condition:str?', 'w'],
            bake: ['烘焙光照,烘焙', 'bake lighting,bake lightmaps', 'Bake {os} for open scenes.', '', 'w'],
        }, [['execute_csharp', '执行 C#,运行一段代码', 'run c# code,execute csharp', 'Compile and run a C# snippet in the Editor context. Escape hatch for anything without a dedicated tool.', 'code:str', 'w']]);
    A('godot', 'Godot', 'game', 'Godot,戈多', 'godot,godot engine', 'Godot 编辑器插件 + GDScript，经 MCP 服务器暴露',
        'Control the Godot editor: scenes, nodes, GDScript, signals and project export.', {
            scene: ['场景', 'scene,scenes', 'open create save run'],
            node: ['节点', 'node,nodes', 'create delete find update'],
            script: ['脚本,GDScript', 'GDScript,GDScripts', 'create attach read'],
            signal: ['信号', 'signal,signals', 'connect list'],
            project: ['工程,项目', 'project,projects', 'run export'],
        }, { connect: ['连接信号,连信号', 'connect signal', 'Connect a {o} to a method.', 'node:str signal:str target:str method:str', 'w'] });
    A('perforce', 'Perforce (Helix Core)', 'game', 'P4,Perforce,版本库,大仓库', 'perforce,p4,helix core', 'p4 命令行 / P4API，经 MCP 服务器暴露',
        'Perforce version control for large game repos: check out, sync, changelists, shelve and submit.', {
            file: ['文件', 'file,files', 'edit add delete revert sync diff history lock'],
            changelist: ['变更列表,CL,changelist', 'changelist,changelists', 'create submit shelve unshelve describe list'],
            stream: ['流,分支,stream', 'stream,streams', 'switch list'],
            workspace: ['工作区,workspace', 'workspace,workspaces', 'create info'],
        }, {
            edit: ['签出,check out,打开编辑', 'check out,open for edit', 'Check out a {o} for editing.', 'path:str changelist:int?', 'w'],
            diff: ['看差异,对比', 'diff', 'Diff the local {o} against the depot.', 'path:str', 'r'],
            history: ['历史,修改记录,谁改的', 'history,filelog,blame', 'Show the revision history of a {o}.', 'path:str limit:int?', 'r'],
            shelve: ['搁置,shelve', 'shelve', 'Shelve the files of a {o}.', 'changelist:int', 'w'],
            unshelve: ['取回搁置', 'unshelve', 'Unshelve a {o} into the workspace.', 'changelist:int', 'w'],
            describe: ['看 CL 详情', 'describe', 'Describe a {o}.', 'changelist:int', 'r'],
            switch: ['切换流,切分支', 'switch stream', 'Switch the workspace to another {o}.', 'stream:str', 'w'],
            info: ['工作区信息', 'workspace info', 'Show information about the {o}.', '', 'r'],
        });
    A('gpuprof', 'GPU 抓帧与性能分析（RenderDoc 等）', 'game', 'RenderDoc,抓帧,GPU 性能,帧分析', 'renderdoc,gpu capture,frame capture,gpu profiler', 'RenderDoc Python API（qrenderdoc）/ 厂商性能工具',
        'Capture and inspect GPU frames: draw calls, pipeline state, textures, shaders and GPU timings.', {
            capture: ['抓帧,截帧,帧捕获', 'capture,captures', 'trigger open list'],
            drawcall: ['Draw Call,绘制调用,DC', 'draw call,draw calls', 'list get'],
            texture: ['贴图,纹理,RT', 'texture,textures', 'get save'],
            shader: ['着色器,Shader', 'shader,shaders', 'get edit_replay'],
            timing: ['GPU 耗时,性能数据,帧耗时', 'gpu timing,gpu timings', 'get compare'],
        }, {
            trigger: ['抓一帧,触发抓帧', 'trigger capture,take a capture', 'Trigger a frame {o} on the connected process.', 'frames:int?', 'w'],
            edit_replay: ['改 Shader 重放,在线改 Shader', 'edit and replay shader', 'Edit a {o} and replay the frame.', 'event_id:int source:str', 'w'],
            compare: ['对比耗时,性能对比', 'compare timings', 'Compare GPU {os} between two captures.', 'a:str b:str', 'r'],
        });
    A('wwise', 'Wwise（音频中间件）', 'game', 'Wwise,音频中间件,游戏音频', 'wwise,audiokinetic', 'WAAPI（WAMP / JSON 接口）',
        'Audiokinetic Wwise authoring via WAAPI: events, sounds, soundbanks and RTPCs.', {
            event: ['音频事件,声音事件', 'event,events', 'create list rename'],
            sound: ['音效,声音', 'sound,sounds', 'import set_property list'],
            soundbank: ['SoundBank,音频包', 'soundbank,soundbanks', 'generate'],
            rtpc: ['RTPC,游戏参数', 'game parameter,game parameters', 'create set'],
        }, { generate: ['生成 SoundBank,出音频包', 'generate soundbanks', 'Generate {os} for platforms.', 'platforms:str[]?', 'w'], set_property: ['改音量,改属性', 'set volume,set property', 'Set a property (Volume, Pitch, …) on a {o}.', 'id:str property:str value:num', 'w'] });

    // —— DCC / 3D / 2D ——
    A('blender', 'Blender', 'dcc', 'Blender,搅拌机,blender', 'blender', 'bpy Python API（Blender 插件开 socket，MCP 服务器转发）',
        'Control Blender: scene objects, modifiers, materials, rendering, import/export and Python.', {
            scene: ['场景', 'scene,scenes', 'get'],
            object: ['物体,对象,模型', 'object,objects', 'create delete duplicate rename set_transform set_parent add_modifier apply_modifier', { create: 'primitive:cube|sphere|cylinder|plane|cone|torus|monkey name:str? location:vec3?' }],
            material: ['材质', 'material,materials', 'create assign set_principled'],
            mesh: ['网格,面数', 'mesh,meshes', 'decimate triangulate uv_unwrap merge_by_distance'],
            render: ['渲染', 'render,renders', 'image animation set_engine set_resolution'],
            file: ['文件,场景文件', 'file,files', 'export_fbx export_gltf export_obj export_usd import_fbx import_gltf import_obj'],
            keyframe: ['关键帧,K 帧', 'keyframe,keyframes', 'insert set_frame_range'],
            asset: ['素材库,PolyHaven 素材,HDRI,贴图素材', 'asset,assets', 'search download'],
            viewport: ['视口', 'viewport,viewports', 'screenshot'],
        }, {
            add_modifier: ['加修改器,添加修改器', 'add modifier', 'Add a modifier (Subdivision, Mirror, Bevel, Array, …) to an {o}.', 'object:str type:SUBSURF|MIRROR|BEVEL|ARRAY|SOLIDIFY|DECIMATE', 'w'],
            apply_modifier: ['应用修改器', 'apply modifier', 'Apply a modifier of an {o}.', 'object:str modifier:str', 'w'],
            set_principled: ['改颜色,粗糙度,金属度,BSDF', 'base color,roughness,metallic,principled bsdf', 'Set Principled BSDF inputs of a {o}.', 'material:str base_color:vec3? roughness:num? metallic:num?', 'w'],
            decimate: ['减面,降面数', 'decimate,reduce polygons', 'Reduce polygon count of a {o}.', 'object:str ratio:num', 'w'],
            triangulate: ['三角化', 'triangulate', 'Triangulate faces of a {o}.', 'object:str', 'w'],
            uv_unwrap: ['展 UV,UV 展开', 'uv unwrap,unwrap uvs', 'Unwrap UVs of a {o}.', 'object:str method:smart|angle_based|lightmap?', 'w'],
            merge_by_distance: ['合并顶点,去重顶点', 'merge by distance,remove doubles', 'Merge vertices of a {o} closer than a distance.', 'object:str distance:num?', 'w'],
            image: ['渲染一张图,出图', 'render image,render still', 'Render the current frame to an image file.', 'output:str samples:int?', 'w'],
            animation: ['渲染动画,出视频', 'render animation', 'Render the frame range to video or image sequence.', 'output:str start:int? end:int?', 'w'],
            set_engine: ['渲染引擎,Cycles,Eevee', 'render engine,cycles,eevee', 'Switch the {o} engine.', 'engine:CYCLES|BLENDER_EEVEE_NEXT|BLENDER_WORKBENCH', 'w'],
            set_resolution: ['分辨率,出图尺寸', 'resolution', 'Set {o} resolution.', 'width:int height:int percentage:int?', 'w'],
            export_fbx: ['导出 FBX,导出成 fbx', 'export fbx,export to fbx', 'Export the scene or selection to FBX.', 'path:str selected_only:bool? apply_modifiers:bool?', 'w'],
            export_gltf: ['导出 glTF,导出 glb', 'export gltf,export glb', 'Export the scene or selection to glTF 2.0 (.glb/.gltf).', 'path:str format:GLB|GLTF_SEPARATE?', 'w'],
            export_obj: ['导出 OBJ', 'export obj', 'Export to Wavefront OBJ.', 'path:str', 'w'],
            export_usd: ['导出 USD', 'export usd', 'Export to USD.', 'path:str', 'w'],
            import_fbx: ['导入 FBX', 'import fbx', 'Import an FBX file.', 'path:str', 'w'],
            import_gltf: ['导入 glTF,导入 glb', 'import gltf,import glb', 'Import a glTF / GLB file.', 'path:str', 'w'],
            import_obj: ['导入 OBJ', 'import obj', 'Import an OBJ file.', 'path:str', 'w'],
            insert: ['插入关键帧,打关键帧', 'insert keyframe', 'Insert a {o} on an object property.', 'object:str data_path:location|rotation_euler|scale frame:int', 'w'],
            set_frame_range: ['帧范围,起止帧', 'frame range', 'Set the scene frame range.', 'start:int end:int', 'w'],
            screenshot: ['截图,看一眼视口', 'screenshot,viewport capture', 'Capture the 3D {o} as an image.', 'max_size:int?', 'r'],
        }, [
            ['get_scene_info', '场景信息,场景里有什么', 'scene info,what is in the scene', 'Get information about the current Blender scene (objects, materials, cameras, lights).', '', 'r'],
            ['execute_blender_code', '执行 Python,运行 bpy 代码,写个脚本', 'run python,execute python code,bpy script', 'Execute arbitrary Python code in Blender (bpy). Escape hatch for anything without a dedicated tool.', 'code:str', 'w'],
        ]);
    A('maya', 'Maya', 'dcc', 'Maya,玛雅', 'maya,autodesk maya', 'maya.cmds / PyMEL Python（commandPort）',
        'Control Autodesk Maya: scene, transforms, rigging, animation, rendering and export.', {
            scene: ['场景,工程文件', 'scene,scenes', 'open save reference'],
            node: ['节点,物体,对象', 'node,nodes', 'create delete rename group set_parent set_transform'],
            joint: ['骨骼,关节', 'joint,joints', 'create bind_skin'],
            animation: ['动画', 'animation,animations', 'set_key bake export_alembic'],
            render: ['渲染', 'render,renders', 'frame set_renderer'],
            file: ['文件', 'file,files', 'export_fbx import_file'],
        }, {
            group: ['打组,成组', 'group', 'Group {os} under a new transform.', 'nodes:str[] name:str', 'w'],
            bind_skin: ['蒙皮,绑定蒙皮', 'bind skin,skin bind', 'Bind a mesh to {os}.', 'mesh:str joints:str[]', 'w'],
            set_key: ['K 帧,设关键帧', 'set key,keyframe', 'Set a keyframe on an attribute.', 'node:str attribute:str time:num value:num?', 'w'],
            export_alembic: ['导出 Alembic,导出 abc', 'export alembic,export abc', 'Export {o} as Alembic cache.', 'nodes:str[] path:str start:int end:int', 'w'],
            frame: ['渲染一帧', 'render frame', 'Render one frame.', 'camera:str output:str', 'w'],
            set_renderer: ['渲染器,Arnold', 'renderer,arnold', 'Set the current renderer.', 'renderer:arnold|mayaHardware2|mayaSoftware', 'w'],
            export_fbx: ['导出 FBX', 'export fbx', 'Export selection to FBX.', 'path:str', 'w'],
            import_file: ['导入文件', 'import file', 'Import a file into the scene.', 'path:str', 'w'],
        }, [['execute_python', '执行 Python,运行 MEL,写个脚本', 'run python,execute mel', 'Execute Python (maya.cmds) or MEL in Maya.', 'code:str language:python|mel?', 'w']]);
    A('houdini', 'Houdini', 'dcc', 'Houdini,胡迪尼,程序化', 'houdini,sidefx houdini', 'hou Python 模块',
        'Control SideFX Houdini: nodes, parameters, simulations, caching and ROP renders.', {
            node: ['节点', 'node,nodes', 'create connect set_parm cook'],
            hda: ['HDA,数字资产', 'digital asset,digital assets', 'install create'],
            simulation: ['模拟,解算', 'simulation,simulations', 'run cache'],
            rop: ['ROP,输出节点', 'ROP,ROPs', 'render'],
        }, {
            connect: ['连线,连接节点', 'connect,wire', 'Connect two {os}.', 'from:str to:str input:int?', 'w'],
            set_parm: ['改参数,设参数', 'set parameter,parm', 'Set a parameter on a {o}.', 'node:str parm:str value:any', 'w'],
            cook: ['Cook,计算节点', 'cook', 'Cook a {o}.', 'node:str', 'w'],
            install: ['安装 HDA', 'install', 'Install a {o} library.', 'path:str', 'w'],
            cache: ['缓存到磁盘,写缓存', 'cache to disk', 'Cache a {o} to disk.', 'node:str path:str start:int end:int', 'w'],
        });
    A('substance', 'Substance 3D Painter', 'dcc', 'Substance,Substance Painter,SP,贴图绘制', 'substance painter,substance 3d painter', 'Painter Python API',
        'Adobe Substance 3D Painter: projects, texture sets, layers, mesh map baking and texture export.', {
            project: ['工程,项目', 'project,projects', 'create open save'],
            texture_set: ['纹理集,Texture Set', 'texture set,texture sets', 'list set_resolution'],
            layer: ['图层,层', 'layer,layers', 'add_fill add_paint add_mask'],
            mesh_map: ['网格贴图,法线贴图,AO', 'mesh map,mesh maps', 'bake'],
            texture: ['贴图,纹理', 'texture,textures', 'export'],
        }, {
            set_resolution: ['改分辨率,贴图尺寸', 'set resolution', 'Set the resolution of a {o}.', 'texture_set:str size:512|1024|2048|4096', 'w'],
            add_fill: ['加填充层', 'add fill layer', 'Add a fill {o}.', 'texture_set:str name:str?', 'w'],
            add_paint: ['加绘画层', 'add paint layer', 'Add a paint {o}.', 'texture_set:str name:str?', 'w'],
            add_mask: ['加遮罩', 'add mask', 'Add a black / white mask to a {o}.', 'layer:str color:black|white?', 'w'],
            bake: ['烘焙贴图,烘法线,烘 AO', 'bake mesh maps,bake normals,bake ao', 'Bake {os} (normal, AO, curvature, …) from high poly.', 'high_poly:str maps:str[]?', 'w'],
            export: ['导出贴图', 'export textures', 'Export {os} with an export preset.', 'preset:str path:str', 'w'],
        });
    A('photoshop', 'Photoshop', 'dcc', 'PS,Photoshop,修图', 'photoshop,ps', 'UXP / ExtendScript 脚本',
        'Adobe Photoshop: documents, layers, selections, adjustments, text, filters, actions and batch export.', {
            document: ['文档,图片,画布', 'document,documents', 'open create save export close resize crop'],
            layer: ['图层', 'layer,layers', 'create delete rename duplicate merge group set_opacity set_blend_mode add_mask set_visible'],
            selection: ['选区', 'selection,selections', 'subject all invert remove_background'],
            adjustment: ['调色,调整图层', 'adjustment,adjustments', 'levels curves hue_saturation'],
            text: ['文字,文本图层', 'text layer,text layers', 'create set_font'],
            filter: ['滤镜', 'filter,filters', 'gaussian_blur sharpen'],
            action: ['动作', 'action,actions', 'run'],
        }, {
            export: ['导出,另存为 PNG,导出 JPG,存成图片', 'export,save as png,export jpg', 'Export the {o} as PNG / JPG / WebP.', 'path:str format:png|jpg|webp quality:int?', 'w'],
            resize: ['改尺寸,缩放图片,调整图像大小', 'resize image,image size', 'Resize the {o}.', 'width:int height:int? resample:bicubic|bilinear|nearest?', 'w'],
            crop: ['裁剪,裁切', 'crop', 'Crop the {o} to a rectangle.', 'left:int top:int right:int bottom:int', 'w'],
            merge: ['合并图层', 'merge layers', 'Merge {os}.', 'layers:str[]', 'w'],
            group: ['图层编组,打组', 'group layers', 'Group {os}.', 'layers:str[] name:str', 'w'],
            set_opacity: ['不透明度,透明度', 'opacity', 'Set the opacity of a {o}.', 'layer:str opacity:num', 'w'],
            set_blend_mode: ['混合模式,叠加,正片叠底', 'blend mode,multiply,overlay', 'Set the blend mode of a {o}.', 'layer:str mode:normal|multiply|screen|overlay|softLight', 'w'],
            add_mask: ['加蒙版,图层蒙版', 'layer mask,add mask', 'Add a mask to a {o}.', 'layer:str', 'w'],
            set_visible: ['显示图层,隐藏图层', 'show layer,hide layer', 'Show or hide a {o}.', 'layer:str visible:bool', 'w'],
            subject: ['选择主体,抠主体', 'select subject', 'Select the main subject (AI).', '', 'w'],
            all: ['全选', 'select all', 'Select the whole canvas.', '', 'w'],
            invert: ['反选', 'invert selection', 'Invert the current {o}.', '', 'w'],
            remove_background: ['去背景,抠图,移除背景', 'remove background', 'Remove the background (AI).', 'layer:str?', 'w'],
            levels: ['色阶', 'levels', 'Apply a Levels {o}.', 'layer:str black:int white:int gamma:num?', 'w'],
            curves: ['曲线', 'curves', 'Apply a Curves {o}.', 'layer:str points:obj', 'w'],
            hue_saturation: ['色相饱和度,饱和度,调色相', 'hue saturation,saturation', 'Apply a Hue/Saturation {o}.', 'layer:str hue:int? saturation:int? lightness:int?', 'w'],
            set_font: ['改字体,字号', 'font,font size', 'Set font, size and color of a {o}.', 'layer:str font:str? size:num? color:vec3?', 'w'],
            gaussian_blur: ['高斯模糊,模糊', 'gaussian blur,blur', 'Apply Gaussian Blur.', 'layer:str radius:num', 'w'],
            sharpen: ['锐化', 'sharpen,unsharp mask', 'Apply Unsharp Mask.', 'layer:str amount:num radius:num?', 'w'],
        }, [['batch_process', '批量处理图片,批处理', 'batch process images,batch', 'Run an action on every image in a folder and save the results.', 'action:str input_folder:str output_folder:str', 'w']]);
    A('figma', 'Figma', 'dcc', 'Figma,设计稿,UI 稿', 'figma,design file', 'Figma REST API / Dev Mode MCP',
        'Figma design files: nodes, frames, components, styles, variables, comments and image export.', {
            file: ['设计文件,稿子', 'file,files', 'get list'],
            node: ['节点,图层,画板', 'node,nodes', 'get export_image'],
            component: ['组件', 'component,components', 'list get create_instance'],
            style: ['样式,设计规范', 'style,styles', 'list'],
            variable: ['变量,设计 Token', 'variable,variables', 'get'],
            comment: ['评论', 'comment,comments', 'create list'],
        }, { export_image: ['导出切图,切图,导出图片', 'export image,export png', 'Export {os} as PNG / SVG / PDF.', 'file:str ids:str[] format:png|svg|pdf scale:num?', 'r'], create_instance: ['用组件,放一个组件', 'create instance', 'Create an instance of a {o}.', 'component:str parent:str', 'w'] });

    // —— 音视频与图片 ——
    A('ffmpeg', 'FFmpeg', 'media', 'FFmpeg,ffmpeg,视频处理', 'ffmpeg', '命令行（经 shell 或专门的 MCP 服务器）',
        'Video and audio processing with FFmpeg: probe, transcode, trim, concat, scale, frames, subtitles and GIFs.', {
            video: ['视频,录像,片子', 'video,videos', 'probe transcode trim concat scale extract_frames change_fps add_subtitles compress make_gif'],
            audio: ['音频,声音,音轨', 'audio,audio files', 'extract convert normalize mix'],
        }, {
            probe: ['看视频信息,分辨率,码率,时长', 'probe,video info,duration,bitrate', 'Print stream information (codec, resolution, duration, bitrate) of a {o}.', 'input:str', 'r'],
            transcode: ['转码,转格式,转成 MP4,转 H.265', 'transcode,convert video,h264,hevc', 'Transcode a {o} to another codec / container.', 'input:str output:str codec:h264|hevc|av1|vp9 crf:int?', 'w'],
            trim: ['剪一段,截取,剪辑,剪掉开头', 'trim,cut,clip', 'Cut a segment out of a {o}.', 'input:str output:str start:str end:str', 'w'],
            concat: ['拼接,合并视频,连起来', 'concatenate,join videos', 'Concatenate several {os}.', 'inputs:str[] output:str', 'w'],
            scale: ['缩放视频,改分辨率,转 720p,转 1080p', 'scale video,resize video,720p,1080p', 'Scale a {o} to a new resolution.', 'input:str output:str width:int height:int?', 'w'],
            extract_frames: ['抽帧,截图,导出每一帧', 'extract frames,thumbnails', 'Extract frames from a {o} as images.', 'input:str output_pattern:str fps:num?', 'w'],
            change_fps: ['改帧率,转 60 帧', 'change frame rate,fps', 'Change the frame rate of a {o}.', 'input:str output:str fps:num', 'w'],
            add_subtitles: ['加字幕,烧字幕', 'add subtitles,burn subtitles', 'Burn subtitles into a {o}.', 'input:str subtitles:str output:str', 'w'],
            compress: ['压缩视频,减小体积', 'compress video,reduce size', 'Re-encode a {o} to a target size or bitrate.', 'input:str output:str target_mb:num?', 'w'],
            make_gif: ['做成 GIF,转动图', 'make gif,convert to gif', 'Convert a {o} clip to an optimized GIF.', 'input:str output:str fps:num? width:int?', 'w'],
            extract: ['提取音频,扒音轨', 'extract audio', 'Extract the {o} track from a video.', 'input:str output:str format:mp3|wav|aac?', 'w'],
            convert: ['转音频格式,转 MP3,转 WAV', 'convert audio,to mp3,to wav', 'Convert {o} to another format.', 'input:str output:str bitrate:str?', 'w'],
            normalize: ['响度标准化,音量统一', 'normalize loudness,loudnorm', 'Normalize loudness of {o} (EBU R128).', 'input:str output:str target_lufs:num?', 'w'],
            mix: ['混音,叠加音乐', 'mix audio', 'Mix several {os} tracks.', 'inputs:str[] output:str', 'w'],
        });
    A('imagemagick', 'ImageMagick', 'media', 'ImageMagick,图片批处理', 'imagemagick,magick', '命令行',
        'Batch image processing: convert, resize, crop, montage/sprite sheets, watermark, optimize.', {
            image: ['图片,图像,贴图,PNG 图', 'image,images,png,texture', 'convert resize crop montage watermark optimize thumbnail'],
        }, {
            montage: ['拼图,合图,做图集,精灵图', 'montage,sprite sheet,atlas', 'Combine {os} into a grid / sprite sheet.', 'inputs:str[] output:str tile:str?', 'w'],
            watermark: ['加水印', 'watermark', 'Overlay a watermark on {os}.', 'input:str watermark:str output:str', 'w'],
            optimize: ['压缩图片,减小体积', 'optimize,compress image', 'Losslessly / lossy optimize {os}.', 'input:str output:str quality:int?', 'w'],
            thumbnail: ['缩略图', 'thumbnail', 'Create thumbnails of {os}.', 'input:str output:str size:int', 'w'],
        });

    // —— 办公与协作 ——
    A('excel', 'Excel', 'office', 'Excel,电子表格,xlsx', 'excel,spreadsheet,xlsx', 'Office JS / openpyxl / COM 自动化',
        'Microsoft Excel workbooks: sheets, ranges, formulas, sorting/filtering, charts, pivot tables and macros.', {
            workbook: ['工作簿,Excel 文件', 'workbook,workbooks', 'open create save'],
            sheet: ['工作表,sheet,表', 'worksheet,worksheets', 'list add delete rename'],
            range: ['单元格,区域,数据', 'range,ranges', 'read write clear sort filter format'],
            formula: ['公式', 'formula,formulas', 'set recalculate'],
            chart: ['图表,折线图,柱状图', 'chart,charts', 'create'],
            pivot: ['数据透视表,透视表', 'pivot table,pivot tables', 'create refresh'],
            data: ['数据', 'data,data', 'remove_duplicates conditional_format text_to_columns'],
            macro: ['宏,VBA', 'macro,macros', 'run'],
        }, {
            read: ['读数据,读取单元格,看看表里', 'read cells,read range', 'Read values of a {o}.', 'sheet:str address:str', 'r'],
            write: ['写数据,填进去,写入单元格', 'write cells,write values', 'Write values into a {o}.', 'sheet:str address:str values:any', 'w'],
            clear: ['清空', 'clear', 'Clear a {o}.', 'sheet:str address:str', 'd'],
            sort: ['排序', 'sort', 'Sort a {o} by a column.', 'sheet:str address:str column:str ascending:bool?', 'w'],
            filter: ['筛选,过滤', 'filter', 'Apply an AutoFilter to a {o}.', 'sheet:str address:str column:str criteria:str', 'w'],
            format: ['格式,加粗,填充颜色,数字格式', 'format,bold,fill color,number format', 'Format a {o} (font, fill, number format).', 'sheet:str address:str format:obj', 'w'],
            set: ['写公式,加公式', 'set formula', 'Set a {o} in a cell.', 'sheet:str cell:str formula:str', 'w'],
            recalculate: ['重新计算', 'recalculate', 'Recalculate all {os}.', '', 'w'],
            refresh: ['刷新透视表', 'refresh pivot', 'Refresh a {o}.', 'name:str', 'w'],
            remove_duplicates: ['删除重复项,去重', 'remove duplicates,dedupe', 'Remove duplicate rows.', 'sheet:str address:str columns:str[]?', 'd'],
            conditional_format: ['条件格式,标红', 'conditional formatting,highlight', 'Add conditional formatting.', 'sheet:str address:str rule:str', 'w'],
            text_to_columns: ['分列', 'text to columns,split column', 'Split a column by a delimiter.', 'sheet:str column:str delimiter:str', 'w'],
        });
    A('word', 'Word', 'office', 'Word,文档,docx', 'word,docx,word document', 'Office JS / python-docx',
        'Microsoft Word documents: paragraphs, find & replace, tables, styles, comments, tracked changes and PDF export.', {
            document: ['文档,Word 文件', 'document,documents', 'open create save export_pdf'],
            text: ['文字,段落', 'paragraph,paragraphs', 'insert find_replace'],
            table: ['表格', 'table,tables', 'insert'],
            comment: ['批注', 'comment,comments', 'add list'],
            revision: ['修订,修改痕迹', 'tracked change,tracked changes', 'accept_all reject_all'],
        }, { export_pdf: ['导出 PDF,转 PDF', 'export pdf,save as pdf', 'Export the {o} as PDF.', 'path:str', 'w'], find_replace: ['查找替换,批量替换', 'find and replace', 'Find and replace text.', 'find:str replace:str match_case:bool?', 'w'], accept_all: ['接受所有修订', 'accept all changes', 'Accept all {os}.', '', 'w'], reject_all: ['拒绝所有修订', 'reject all changes', 'Reject all {os}.', '', 'w'] });
    A('powerpoint', 'PowerPoint', 'office', 'PPT,幻灯片,演示文稿,PowerPoint', 'powerpoint,ppt,slides', 'Office JS / python-pptx',
        'Microsoft PowerPoint: slides, layouts, text boxes, images, charts, speaker notes, themes and PDF export.', {
            presentation: ['演示文稿,PPT', 'presentation,presentations', 'open create save export_pdf'],
            slide: ['幻灯片,页', 'slide,slides', 'add delete reorder set_layout'],
            shape: ['文本框,图片,形状', 'shape,shapes', 'add_text add_image add_chart'],
            notes: ['备注,演讲者备注', 'speaker note,speaker notes', 'set'],
            theme: ['主题,模板', 'theme,themes', 'apply'],
        }, { export_pdf: ['导出 PDF', 'export pdf', 'Export the {o} as PDF.', 'path:str', 'w'], reorder: ['调整顺序,换页序', 'reorder slides', 'Move a {o} to another position.', 'index:int to:int', 'w'], set_layout: ['版式,换版式', 'layout', 'Change the layout of a {o}.', 'index:int layout:str', 'w'], add_text: ['加文本框,加文字', 'add text box', 'Add a text box.', 'slide:int text:str', 'w'], add_image: ['加图片,插入图片', 'add image,insert picture', 'Insert an image.', 'slide:int path:str', 'w'], add_chart: ['加图表,插入图表', 'add chart', 'Insert a chart.', 'slide:int type:bar|line|pie data:obj', 'w'], set: ['写备注', 'set notes', 'Set {os} of a slide.', 'slide:int text:str', 'w'] });
    A('mail', '邮件（Outlook / Gmail）', 'office', '邮箱,Outlook,Gmail', 'email,mail,outlook,gmail', 'Microsoft Graph / Gmail API',
        'Email: search, read, send, reply, forward, folders, attachments and rules.', {
            message: ['邮件,信', 'email,emails', 'search get send reply forward move mark_read'],
            attachment: ['附件', 'attachment,attachments', 'download'],
            folder: ['文件夹,收件箱', 'mail folder,mail folders', 'list create'],
            rule: ['收信规则,过滤规则', 'mail rule,mail rules', 'create'],
        }, { forward: ['转发', 'forward', 'Forward an {o}.', 'id:str to:str[] comment:str?', 'o'], mark_read: ['标为已读', 'mark as read', 'Mark {os} as read.', 'ids:str[]', 'w'], send: ['发邮件,写邮件,发一封', 'send email,email someone', 'Send an {o}.', 'to:str[] subject:str body:str cc:str[]?', 'o'] });
    A('calendar', '日历（Outlook / Google Calendar）', 'office', '日历,Outlook 日历,Google 日历', 'calendar,meeting,schedule', 'Microsoft Graph / Google Calendar API',
        'Calendars: list, create, update and cancel events, find free time and book rooms.', {
            event: ['日程,会议,活动', 'event,events', 'list create update delete find_free_time'],
            room: ['会议室', 'meeting room,meeting rooms', 'book list'],
        }, { find_free_time: ['找空闲时间,大家什么时候有空', 'find free time,availability', 'Find common free slots for attendees.', 'attendees:str[] duration_min:int range_start:str range_end:str', 'r'], book: ['订会议室,约会议室', 'book room', 'Book a {o}.', 'room:str start:str end:str', 'o'] });
    A('feishu', '飞书（Lark）', 'office', '飞书,Lark', 'feishu,lark', '飞书开放平台 OpenAPI',
        'Feishu / Lark: docs, sheets, Bitable, messages, group chats, calendar, approvals and wiki.', {
            doc: ['云文档,飞书文档,文档', 'doc,docs', 'create get append search share'],
            sheet: ['电子表格,飞书表格', 'sheet,sheets', 'read write add'],
            bitable: ['多维表格,数据表', 'bitable record,bitable records', 'list create update search'],
            message: ['消息,飞书消息', 'message,messages', 'send search reply'],
            chat: ['群,群聊', 'group chat,group chats', 'create invite'],
            event: ['日程,会议', 'calendar event,calendar events', 'create list'],
            approval: ['审批,审批单', 'approval,approvals', 'create approve reject list'],
            wiki: ['知识库,Wiki', 'wiki node,wiki nodes', 'list move'],
        }, { append: ['追加内容,往文档里写', 'append content', 'Append blocks to a {o}.', 'doc:str markdown:str', 'w'], read: ['读表格', 'read sheet', 'Read a range of a {o}.', 'sheet:str range:str', 'r'], write: ['写表格', 'write sheet', 'Write a range of a {o}.', 'sheet:str range:str values:any', 'w'] });
    A('notion', 'Notion', 'office', 'Notion,笔记', 'notion', 'Notion API',
        'Notion: pages, databases, blocks and comments.', {
            page: ['页面,笔记', 'page,pages', 'create get update search'],
            database: ['数据库,表格', 'database,databases', 'query create_item update_item'],
            block: ['块,内容块', 'block,blocks', 'append'],
            comment: ['评论', 'comment,comments', 'create'],
        }, { create_item: ['加一行,新建条目', 'add row,create item', 'Create an item in a {o}.', 'database:str properties:obj', 'w'], update_item: ['改条目', 'update item', 'Update an item in a {o}.', 'page:str properties:obj', 'w'], append: ['追加内容', 'append blocks', 'Append {os} to a page.', 'page:str markdown:str', 'w'] });
    A('gdrive', 'Google Drive / 网盘', 'office', '网盘,云盘,Google Drive,共享盘', 'google drive,drive,cloud drive', 'Google Drive API',
        'Cloud drive files: search, upload, download, share, move, folders and export.', {
            file: ['文件,网盘文件', 'file,files', 'search upload download share move delete export_pdf'],
            folder: ['文件夹,目录', 'folder,folders', 'create list'],
        }, { export_pdf: ['导出 PDF', 'export as pdf', 'Export a Google Docs {o} as PDF.', 'id:str path:str', 'r'] });
    A('confluence', 'Confluence', 'office', 'Confluence,Wiki,文档站', 'confluence,wiki', 'Confluence REST API',
        'Confluence wiki: pages, spaces, labels and comments.', {
            page: ['Wiki 页面,文档页', 'page,pages', 'create get update search move'],
            space: ['空间', 'space,spaces', 'list'],
            label: ['标签', 'label,labels', 'add'],
        }, { add: ['加标签,打标签', 'add label', 'Add a {o} to a page.', 'page:str label:str', 'w'] });

    // —— 开发与运维 ——
    A('filesystem', '文件系统', 'dev', '本地,本地磁盘,文件系统', 'file,files,filesystem,directory', '参考 MCP 服务器（filesystem）', 'Local filesystem access restricted to allowed directories.', {}, {}, [
        ['read_file', '读文件,打开文件看看,看文件内容', 'read file,open file,show file contents', 'Read the complete contents of a file as text.', 'path:str', 'r'],
        ['read_multiple_files', '一次读多个文件', 'read several files', 'Read several files at once.', 'paths:str[]', 'r'],
        ['write_file', '写文件,新建文件,保存成文件', 'write file,create file,save to file', 'Create a new file or completely overwrite an existing file.', 'path:str content:str', 'd'],
        ['edit_file', '改文件,替换文件里的,修改代码', 'edit file,replace text in file,patch file', 'Make line-based edits to a text file (find exact text → replace).', 'path:str edits:obj dry_run:bool?', 'w'],
        ['create_directory', '新建文件夹,建目录', 'create directory,mkdir,new folder', 'Create a new directory (with parents).', 'path:str', 'w'],
        ['list_directory', '列目录,看文件夹里有什么,ls', 'list directory,ls,folder contents', 'List files and directories in a path.', 'path:str', 'r'],
        ['directory_tree', '目录树,文件结构', 'directory tree,folder structure', 'Get a recursive tree view of files and directories as JSON.', 'path:str', 'r'],
        ['move_file', '移动文件,重命名文件', 'move file,rename file', 'Move or rename a file or directory.', 'source:str destination:str', 'w'],
        ['search_files', '找文件,搜文件名', 'find files,search file names', 'Recursively search for files matching a pattern.', 'path:str pattern:str', 'r'],
        ['get_file_info', '文件信息,文件大小,修改时间', 'file info,file size,modified time', 'Retrieve metadata about a file (size, times, permissions).', 'path:str', 'r'],
    ]);
    A('git', 'Git', 'dev', 'git,本地仓库', 'git,repository,repo', '参考 MCP 服务器（git）/ 命令行', 'Local Git repository operations.', {}, {}, [
        ['git_status', '看状态,改了哪些文件', 'git status,what changed', 'Show the working tree status.', 'repo_path:str', 'r'],
        ['git_diff_unstaged', '看改动,未暂存的修改,diff', 'git diff,unstaged changes', 'Show changes in the working directory that are not yet staged.', 'repo_path:str', 'r'],
        ['git_diff_staged', '看暂存的修改', 'staged diff,diff cached', 'Show changes that are staged for commit.', 'repo_path:str', 'r'],
        ['git_add', '暂存,git add', 'git add,stage files', 'Add file contents to the staging area.', 'repo_path:str files:str[]', 'w'],
        ['git_commit', '提交,git commit', 'git commit,commit changes', 'Record staged changes to the repository.', 'repo_path:str message:str', 'w'],
        ['git_log', '提交历史,git log,最近的提交', 'git log,commit history', 'Show the commit logs.', 'repo_path:str max_count:int?', 'r'],
        ['git_show', '看某次提交', 'git show,show commit', 'Show the contents of a commit.', 'repo_path:str revision:str', 'r'],
        ['git_create_branch', '新建分支,开分支', 'create branch,new branch', 'Create a new branch from an optional base.', 'repo_path:str branch_name:str base_branch:str?', 'w'],
        ['git_checkout', '切换分支,checkout', 'checkout,switch branch', 'Switch branches.', 'repo_path:str branch_name:str', 'w'],
        ['git_reset', '取消暂存,reset', 'git reset,unstage', 'Unstage all staged changes.', 'repo_path:str', 'w'],
        ['git_blame', '谁改的这一行,blame', 'git blame,who changed this line', 'Show what revision and author last modified each line of a file.', 'repo_path:str path:str', 'r'],
        ['git_stash', '暂存修改到 stash,stash', 'git stash', 'Stash the changes in a dirty working directory.', 'repo_path:str message:str?', 'w'],
    ]);
    A('github', 'GitHub', 'dev', 'GitHub,GitHub 上', 'github,pull request,pr,issue', '官方 GitHub MCP 服务器 / REST API', 'GitHub repositories, issues, pull requests, code search and Actions.', {}, {}, [
        ['create_issue', '提 issue,新建 issue,报 bug', 'create issue,open issue,file a bug', 'Create a new issue in a repository.', 'owner:str repo:str title:str body:str? labels:str[]?', 'o'],
        ['list_issues', '列出 issue,看看有哪些 issue', 'list issues', 'List issues in a repository with filters.', 'owner:str repo:str state:open|closed|all? labels:str[]?', 'r'],
        ['update_issue', '改 issue,关闭 issue', 'update issue,close issue', 'Update an existing issue.', 'owner:str repo:str issue_number:int state:open|closed? title:str?', 'o'],
        ['add_issue_comment', '评论 issue,回复 issue', 'comment on issue', 'Add a comment to an issue.', 'owner:str repo:str issue_number:int body:str', 'o'],
        ['search_issues', '搜 issue', 'search issues', 'Search issues and pull requests across GitHub.', 'q:str', 'r'],
        ['create_pull_request', '提 PR,开 PR,发合并请求', 'create pull request,open pr', 'Create a new pull request.', 'owner:str repo:str title:str head:str base:str body:str?', 'o'],
        ['list_pull_requests', '列出 PR,看看有哪些 PR', 'list pull requests', 'List pull requests in a repository.', 'owner:str repo:str state:open|closed|all?', 'r'],
        ['get_pull_request_diff', '看 PR 改动,PR diff', 'pr diff,pull request changes', 'Get the diff of a pull request.', 'owner:str repo:str pull_number:int', 'r'],
        ['create_pull_request_review', '审 PR,review,批准 PR', 'review pr,approve pr', 'Create a review on a pull request.', 'owner:str repo:str pull_number:int event:APPROVE|REQUEST_CHANGES|COMMENT body:str?', 'o'],
        ['merge_pull_request', '合并 PR,合进去', 'merge pr,merge pull request', 'Merge a pull request.', 'owner:str repo:str pull_number:int merge_method:merge|squash|rebase?', 'd'],
        ['get_file_contents', '看仓库里的文件,远程文件', 'get file contents,file in repo', 'Get the contents of a file or directory from a repository.', 'owner:str repo:str path:str ref:str?', 'r'],
        ['create_or_update_file', '在仓库里改文件,直接推一个文件', 'create or update file in repo', 'Create or update a single file in a repository.', 'owner:str repo:str path:str content:str message:str branch:str', 'o'],
        ['push_files', '推送多个文件', 'push files', 'Push multiple files in a single commit.', 'owner:str repo:str branch:str files:obj message:str', 'o'],
        ['create_branch', '在 GitHub 上建分支', 'create remote branch', 'Create a new branch in a repository.', 'owner:str repo:str branch:str from_branch:str?', 'o'],
        ['search_code', '搜代码,在 GitHub 上搜代码', 'search code', 'Search for code across GitHub repositories.', 'q:str', 'r'],
        ['search_repositories', '搜仓库,找开源项目', 'search repositories,find repos', 'Search for GitHub repositories.', 'query:str', 'r'],
        ['list_commits', '看提交记录,远程提交', 'list commits', 'Get the list of commits of a branch.', 'owner:str repo:str sha:str?', 'r'],
        ['fork_repository', 'fork 仓库', 'fork repository', 'Fork a repository.', 'owner:str repo:str', 'o'],
        ['list_workflow_runs', 'CI 状态,Actions 跑了没,流水线', 'workflow runs,ci status,github actions', 'List GitHub Actions workflow runs.', 'owner:str repo:str status:str?', 'r'],
        ['rerun_workflow', '重跑 CI,重新跑流水线', 'rerun workflow,rerun ci', 'Re-run a failed workflow run.', 'owner:str repo:str run_id:int', 'o'],
    ]);
    A('shell', '终端 / Shell', 'dev', '终端,命令行,shell,cmd,PowerShell', 'shell,terminal,bash,command line', 'Bash / PowerShell（本地或沙箱）', 'Run shell commands in a sandboxed terminal.', {}, {}, [
        ['run_command', '运行命令,执行命令,在终端里跑', 'run command,execute shell command,bash', 'Run a shell command and return stdout / stderr / exit code. Use for CLIs (git, npm, compilers, ffmpeg, …).', 'command:str cwd:str? timeout_s:int?', 'w'],
        ['run_background', '后台运行,起个服务', 'run in background,start server', 'Start a long-running command in the background and return a handle.', 'command:str cwd:str?', 'w'],
        ['read_output', '看后台输出', 'read background output', 'Read new output of a background command.', 'handle:str', 'r'],
        ['kill_process', '杀进程,结束进程', 'kill process', 'Terminate a process or background command.', 'handle:str', 'd'],
        ['list_processes', '看进程,哪些进程在跑', 'list processes,ps', 'List running processes.', 'filter:str?', 'r'],
    ]);
    A('runtime', '代码执行（Python / Node / 测试）', 'dev', 'Python,Node,运行代码,单元测试', 'python,node,run code,unit tests', '沙箱解释器', 'Execute code in a sandbox and run test suites.', {}, {}, [
        ['python_execute', '跑 Python,执行 Python 代码,算一下', 'run python,execute python,calculate', 'Execute Python code in a sandbox and return stdout and the last expression.', 'code:str timeout_s:int?', 'w'],
        ['node_execute', '跑 JS,执行 JavaScript', 'run javascript,execute node', 'Execute JavaScript (Node.js) in a sandbox.', 'code:str', 'w'],
        ['pip_install', '装 Python 包,pip install', 'pip install,install python package', 'Install Python packages into the sandbox.', 'packages:str[]', 'w'],
        ['npm_install', '装依赖,npm install', 'npm install,install dependencies', 'Install npm dependencies.', 'cwd:str packages:str[]?', 'w'],
        ['run_tests', '跑测试,单元测试,跑一下 test', 'run tests,unit tests,pytest,jest', 'Run the project test suite and report failures.', 'cwd:str filter:str?', 'w'],
        ['lint', '代码检查,lint,格式化', 'lint,format code', 'Run linters / formatters on files.', 'paths:str[] fix:bool?', 'w'],
    ]);
    A('docker', 'Docker', 'dev', 'Docker,容器,镜像', 'docker,container,image', 'Docker Engine API / CLI', 'Docker containers, images, compose stacks and volumes.', {
        container: ['容器', 'container,containers', 'list run stop delete logs exec'],
        image: ['镜像', 'image,images', 'build pull push list delete'],
        compose: ['compose,编排', 'compose stack,compose stacks', 'up down ps'],
        volume: ['数据卷,卷', 'volume,volumes', 'list prune'],
    }, { exec: ['进容器执行,在容器里跑命令', 'exec in container', 'Execute a command inside a running {o}.', 'id:str command:str', 'w'], pull: ['拉镜像', 'pull image', 'Pull an {o} from a registry.', 'name:str tag:str?', 'w'], push: ['推镜像', 'push image', 'Push an {o} to a registry.', 'name:str tag:str', 'o'], up: ['起服务,docker compose up', 'compose up', 'Start a {o}.', 'path:str', 'w'], down: ['停服务,compose down', 'compose down', 'Stop a {o}.', 'path:str', 'w'], ps: ['看服务状态', 'compose ps', 'List services of a {o}.', 'path:str', 'r'], prune: ['清理无用卷', 'prune volumes', 'Remove unused {os}.', '', 'd'] });
    A('k8s', 'Kubernetes', 'dev', 'K8s,Kubernetes,集群,pod', 'kubernetes,k8s,cluster', 'Kubernetes API / kubectl', 'Kubernetes clusters: pods, deployments, services, config and events.', {
        pod: ['Pod,容器组', 'pod,pods', 'list get logs delete exec'],
        deployment: ['Deployment,部署,服务', 'deployment,deployments', 'list scale restart rollout_status set_image rollback'],
        service: ['Service,服务入口', 'service,services', 'list port_forward'],
        configmap: ['ConfigMap,配置', 'configmap,configmaps', 'get apply'],
        event: ['事件,集群事件', 'cluster event,cluster events', 'list'],
    }, { scale: ['扩容,缩容,调副本数', 'scale,replicas', 'Scale a {o} to N replicas.', 'name:str namespace:str replicas:int', 'w'], rollout_status: ['发布状态,滚动更新进度', 'rollout status', 'Show rollout status of a {o}.', 'name:str namespace:str', 'r'], set_image: ['换镜像,升级版本', 'set image,update image', 'Update the container image of a {o}.', 'name:str namespace:str image:str', 'w'], rollback: ['回滚', 'rollback,undo rollout', 'Roll back a {o} to the previous revision.', 'name:str namespace:str', 'w'], port_forward: ['端口转发', 'port forward', 'Forward a local port to a {o}.', 'name:str namespace:str port:int', 'w'], apply: ['应用配置', 'apply', 'Apply a {o} manifest.', 'manifest:str', 'w'], exec: ['进 Pod 执行', 'exec in pod', 'Execute a command in a {o}.', 'name:str namespace:str command:str', 'w'] });
    A('aws', 'AWS', 'dev', 'AWS,亚马逊云,S3,EC2,Lambda', 'aws,amazon web services,s3,ec2,lambda', 'AWS SDK / CLI', 'AWS: S3 objects, EC2 instances, Lambda functions, CloudWatch logs and metrics.', {
        bucket: ['S3 桶,存储桶', 'S3 bucket,S3 buckets', 'list'],
        object: ['S3 文件,对象存储文件', 'S3 object,S3 objects', 'list upload download delete presign'],
        instance: ['EC2,云服务器,实例', 'EC2 instance,EC2 instances', 'list start stop reboot get'],
        function: ['Lambda,云函数', 'Lambda function,Lambda functions', 'list invoke update_code logs'],
        log: ['CloudWatch 日志,云日志', 'CloudWatch log,CloudWatch logs', 'query'],
        metric: ['CloudWatch 指标,监控指标', 'CloudWatch metric,CloudWatch metrics', 'get'],
    }, {});
    A('sql', '数据库（PostgreSQL / MySQL）', 'dev', '数据库,SQL,Postgres,MySQL,数据表', 'database,sql,postgres,mysql', '数据库驱动（只读账号 + 写操作需确认）', 'Relational databases: run read-only queries, inspect schemas, explain plans.', {}, {}, [
        ['query', '查数据库,跑 SQL,查表', 'run sql,query database,select', 'Run a read-only SQL query and return rows (max 500).', 'sql:str', 'r'],
        ['execute', '改数据,执行 SQL 写入,update 表', 'execute sql,write to database,update rows', 'Execute a write SQL statement (INSERT/UPDATE/DELETE). Requires confirmation.', 'sql:str', 'd'],
        ['list_tables', '有哪些表,列出数据表', 'list tables,show tables', 'List tables in the database.', 'schema:str?', 'r'],
        ['describe_table', '表结构,字段有哪些', 'describe table,table schema,columns', 'Show columns, types and indexes of a table.', 'table:str', 'r'],
        ['explain_query', '慢查询分析,执行计划,explain', 'explain query,query plan,slow query', 'Show the execution plan of a query.', 'sql:str', 'r'],
    ]);
    A('redis', 'Redis', 'dev', 'Redis,缓存', 'redis,cache', 'Redis 客户端', 'Redis keys: get, set, delete, scan and TTL.', {}, {}, [
        ['redis_get', '读缓存,取 key', 'get key', 'Get the value of a key.', 'key:str', 'r'],
        ['redis_set', '写缓存,设置 key', 'set key', 'Set a key with optional expiry.', 'key:str value:str ttl_s:int?', 'w'],
        ['redis_delete', '删缓存,删 key', 'delete key', 'Delete keys.', 'keys:str[]', 'd'],
        ['redis_scan', '扫 key,找 key', 'scan keys', 'Iterate keys matching a pattern.', 'pattern:str count:int?', 'r'],
        ['redis_ttl', '过期时间,TTL', 'ttl,expiry', 'Get the remaining time to live of a key.', 'key:str', 'r'],
    ]);
    A('sentry', 'Sentry', 'dev', 'Sentry,崩溃,报错监控,线上错误', 'sentry,crash,error monitoring', 'Sentry API', 'Error monitoring: issues, events, stack traces and releases.', {
        issue: ['错误,崩溃,问题', 'issue,issues', 'list get resolve assign'],
        event: ['错误事件,堆栈', 'event,events', 'get_latest'],
        release: ['版本,发布版本', 'release,releases', 'create'],
    }, { resolve: ['标记已解决', 'resolve', 'Mark an {o} as resolved.', 'id:str', 'w'], get_latest: ['最新堆栈,最近一次报错', 'latest event,stack trace', 'Get the latest {o} of an issue with stack trace.', 'issue_id:str', 'r'] });
    A('jira', 'Jira', 'dev', 'Jira,工单,任务单,需求单', 'jira,ticket,task', 'Jira REST API', 'Jira issues, transitions, comments, JQL search and sprints.', {
        issue: ['Jira 单,任务,工单,bug 单', 'issue,issues', 'create get update transition assign comment search link'],
        sprint: ['迭代,Sprint', 'sprint,sprints', 'list start complete'],
    }, { transition: ['流转,改状态,移到完成', 'transition,move to done,change status', 'Transition an {o} to another status.', 'key:str status:str', 'w'], search: ['JQL 查询,搜工单', 'jql,search issues', 'Search {os} with JQL.', 'jql:str limit:int?', 'r'], link: ['关联工单', 'link issues', 'Link two {os}.', 'from:str to:str type:blocks|relates|duplicates', 'w'], complete: ['结束迭代', 'complete sprint', 'Complete a {o}.', 'id:str', 'w'] });
    A('browser', '浏览器自动化（Playwright）', 'dev', '浏览器,Playwright', 'browser,web page,playwright', '官方 Playwright MCP 服务器', 'Drive a real browser: navigate, click, type, read the accessibility snapshot, screenshots.', {}, {}, [
        ['browser_navigate', '打开网页,访问网址', 'open url,navigate,go to page', 'Navigate to a URL.', 'url:str', 'r'],
        ['browser_click', '点击网页按钮,点一下', 'click,click button', 'Click an element on the page (by accessibility ref).', 'element:str ref:str', 'w'],
        ['browser_type', '在网页输入框里输入,填表', 'type into field,fill input', 'Type text into an editable element.', 'element:str ref:str text:str submit:bool?', 'w'],
        ['browser_snapshot', '读网页结构,网页快照', 'page snapshot,accessibility tree', 'Capture an accessibility snapshot of the current page (better than a screenshot for acting).', '', 'r'],
        ['browser_take_screenshot', '网页截图', 'web screenshot', 'Take a screenshot of the current page.', 'full_page:bool?', 'r'],
        ['browser_select_option', '下拉框选择', 'select option,dropdown', 'Select an option in a dropdown.', 'element:str ref:str values:str[]', 'w'],
        ['browser_press_key', '按键,回车', 'press key,enter', 'Press a key on the keyboard.', 'key:str', 'w'],
        ['browser_wait_for', '等页面加载,等文字出现', 'wait for,wait until', 'Wait for text to appear / disappear or for a time.', 'text:str? time_s:num?', 'r'],
        ['browser_evaluate', '在网页里执行 JS', 'evaluate javascript', 'Evaluate JavaScript on the page.', 'function:str', 'w'],
    ]);
    A('web', '联网搜索与抓取', 'dev', '网上,百度,谷歌', 'web search,google,search the web,fetch url', '搜索 API / HTTP 抓取', 'Search the web and fetch pages as markdown.', {}, {}, [
        ['web_search', '上网搜,搜一下,查资料', 'search the web,google it,look up online', 'Search the web and return result titles, URLs and snippets.', 'query:str', 'r'],
        ['web_fetch', '抓网页,读网页内容,打开链接看看', 'fetch url,read web page', 'Fetch a URL and convert the page to markdown.', 'url:str prompt:str?', 'r'],
    ]);
    A('slack', 'Slack', 'office', 'Slack,频道', 'slack,channel', '参考 MCP 服务器（slack）', 'Slack workspace: channels, messages, threads, reactions and users.', {}, {}, [
        ['slack_list_channels', '有哪些频道', 'list channels', 'List public channels in the workspace.', 'limit:int? cursor:str?', 'r'],
        ['slack_post_message', '发 Slack 消息,在频道里说', 'post message,send slack message', 'Post a new message to a channel.', 'channel_id:str text:str', 'o'],
        ['slack_reply_to_thread', '回复帖子,回复线程', 'reply in thread', 'Reply to a specific message thread.', 'channel_id:str thread_ts:str text:str', 'o'],
        ['slack_add_reaction', '点表情,加 reaction', 'add reaction,emoji', 'Add a reaction emoji to a message.', 'channel_id:str timestamp:str reaction:str', 'o'],
        ['slack_get_channel_history', '频道聊天记录,看看频道里说了啥', 'channel history,recent messages', 'Get recent messages from a channel.', 'channel_id:str limit:int?', 'r'],
        ['slack_get_thread_replies', '看线程回复', 'thread replies', 'Get all replies in a message thread.', 'channel_id:str thread_ts:str', 'r'],
        ['slack_get_users', '有哪些成员', 'list users', 'Get a list of workspace users.', 'limit:int? cursor:str?', 'r'],
        ['slack_get_user_profile', '查某人资料', 'user profile', 'Get detailed profile information for a user.', 'user_id:str', 'r'],
    ]);

    // —— 数据与运营 ——
    A('analytics', '运营数据分析', 'data', '运营数据,数据分析,报表,DAU,留存', 'analytics,dashboard,metrics,dau,retention', '数据平台查询 API', 'Game / product analytics: DAU, retention, revenue, ARPU, LTV, funnels, cohorts, A/B tests and dashboards.', {
        metric: ['指标,数据', 'metric,metrics', 'dau retention revenue arpu ltv'],
        funnel: ['漏斗,转化', 'funnel,funnels', 'create'],
        cohort: ['同期群,分群', 'cohort,cohorts', 'analyze'],
        experiment: ['AB 实验,A/B 测试', 'A/B experiment,A/B experiments', 'create results'],
        dashboard: ['看板,报表', 'dashboard,dashboards', 'create export'],
        events: ['埋点,事件日志', 'event log,event logs', 'query'],
    }, {
        dau: ['DAU,日活,活跃用户', 'dau,daily active users', 'Query daily active users for a date range, optionally grouped by channel or region.', 'app:str start_date:str end_date:str group_by:channel|region|platform?', 'r'],
        retention: ['留存,次留,七留', 'retention,day-1 retention,d7', 'Query N-day retention for new users.', 'app:str start_date:str end_date:str days:int[]?', 'r'],
        revenue: ['收入,流水,付费', 'revenue,sales,income', 'Query revenue / paying users for a date range.', 'app:str start_date:str end_date:str currency:str?', 'r'],
        arpu: ['ARPU,ARPPU,人均付费', 'arpu,arppu', 'Query ARPU / ARPPU.', 'app:str start_date:str end_date:str', 'r'],
        ltv: ['LTV,生命周期价值,回本', 'ltv,lifetime value,roi', 'Query LTV curves per install cohort.', 'app:str cohort_start:str days:int', 'r'],
        analyze: ['分群分析,同期群分析', 'cohort analysis', 'Run a {o} analysis.', 'app:str definition:obj', 'r'],
        results: ['实验结果,哪个版本好', 'experiment results', 'Get statistical results of an {o}.', 'id:str', 'r'],
        export: ['导出报表', 'export report', 'Export a {o} as CSV / PDF.', 'id:str format:csv|pdf', 'r'],
    });
    A('support', '客服工单', 'data', '客服,工单,投诉,玩家反馈', 'support,ticket,customer service,complaint', '客服系统 API', 'Customer support tickets: search, read, reply, tag, assign and close.', {
        ticket: ['工单,客服单,投诉单', 'ticket,tickets', 'search get reply assign close tag'],
    }, { close: ['关单,关闭工单', 'close ticket', 'Close a {o}.', 'id:str resolution:str?', 'w'], tag: ['打标签', 'tag ticket', 'Add tags to a {o}.', 'id:str tags:str[]', 'w'] });

    // —— 常用商业软件与游戏服务（SaaS 长尾）——
    const S = (ns, name, domain, zh, en, objects, verbs) => A(ns, name, domain, zh, en, 'REST API（OAuth），经厂商或社区的 MCP 服务器暴露', `${name}: ${Object.values(objects).map((o) => o[1].split(',')[1]).join(', ')}.`, objects, verbs);
    const CRUD = 'create get update delete list search';
    S('salesforce', 'Salesforce', 'crm', 'Salesforce,SFDC', 'salesforce', { lead: ['线索,潜在客户', 'lead,leads', CRUD + ' convert'], contact: ['联系人', 'contact,contacts', CRUD], account: ['客户,公司', 'account,accounts', CRUD], opportunity: ['商机,机会', 'opportunity,opportunities', CRUD], case: ['客诉,服务请求', 'case,cases', 'create get update list'] }, { convert: ['转化线索,转成客户', 'convert lead', 'Convert a {o} into an account, contact and opportunity.', 'id:str', 'w'] });
    S('hubspot', 'HubSpot', 'crm', 'HubSpot', 'hubspot', { contact: ['联系人,客户', 'contact,contacts', CRUD], company: ['公司', 'company,companies', CRUD], deal: ['交易,商机,单子', 'deal,deals', CRUD], ticket: ['工单', 'ticket,tickets', 'create get update list'], note: ['备注,跟进记录', 'note,notes', 'create list'] });
    S('zendesk', 'Zendesk', 'support', 'Zendesk', 'zendesk', { ticket: ['工单,客服单', 'ticket,tickets', CRUD + ' assign'], user: ['用户,客户', 'user,users', 'create get update search'], macro: ['宏,快捷回复', 'macro,macros', 'list apply'], article: ['帮助文章,知识库文章', 'help center article,help center articles', 'create update search'] });
    S('freshdesk', 'Freshdesk', 'support', 'Freshdesk', 'freshdesk', { ticket: ['工单', 'ticket,tickets', CRUD + ' reply'], contact: ['联系人', 'contact,contacts', 'create get update list'] });
    S('intercom', 'Intercom', 'support', 'Intercom', 'intercom', { conversation: ['会话,对话', 'conversation,conversations', 'list get reply close assign'], contact: ['用户,联系人', 'contact,contacts', 'create get update search'] }, { close: ['关闭会话', 'close conversation', 'Close a {o}.', 'id:str', 'w'] });
    S('asana', 'Asana', 'pm', 'Asana', 'asana', { task: ['任务,待办', 'task,tasks', CRUD + ' assign complete'], project: ['项目', 'project,projects', 'create get update list'], section: ['分组,栏目', 'section,sections', 'create list'], comment: ['评论', 'comment,comments', 'create list'] });
    S('trello', 'Trello', 'pm', 'Trello', 'trello', { card: ['卡片', 'card,cards', CRUD + ' move'], board: ['看板,面板', 'board,boards', 'create get list'], list: ['列表,栏', 'list,lists', 'create archive'], checklist: ['检查清单', 'checklist,checklists', 'create'] });
    S('clickup', 'ClickUp', 'pm', 'ClickUp', 'clickup', { task: ['任务', 'task,tasks', CRUD + ' assign'], list: ['列表', 'list,lists', 'create get'], goal: ['目标', 'goal,goals', 'create update'], time_entry: ['工时,计时', 'time entry,time entries', 'create list'] });
    S('linear', 'Linear', 'pm', 'Linear', 'linear', { issue: ['问题单,任务,issue', 'issue,issues', CRUD + ' assign'], project: ['项目', 'project,projects', 'create get list update'], cycle: ['周期,迭代', 'cycle,cycles', 'list get'] });
    S('monday', 'monday.com', 'pm', 'Monday,monday.com', 'monday,monday.com', { item: ['条目,事项', 'item,items', CRUD], board: ['看板,表', 'board,boards', 'create list get'], update: ['动态,进展更新', 'update post,update posts', 'create list'] });
    S('tapd', 'TAPD', 'pm', 'TAPD,腾讯 TAPD', 'tapd', { story: ['需求', 'story,stories', CRUD], bug: ['缺陷,bug 单', 'bug,bugs', CRUD + ' assign'], task: ['任务', 'task,tasks', CRUD], iteration: ['迭代', 'iteration,iterations', 'create list get'], release: ['发布计划,版本', 'release plan,release plans', 'create list'] });
    S('zentao', '禅道', 'pm', '禅道,ZenTao', 'zentao', { story: ['需求', 'story,stories', 'create get update list'], bug: ['Bug,缺陷', 'bug,bugs', 'create get update list assign'], testcase: ['用例,测试用例', 'test case,test cases', 'create list run'], build: ['版本,构建版本', 'build,builds', 'create list'] }, { run: ['执行用例,跑用例', 'run test case', 'Run a {o} and record the result.', 'id:str result:pass|fail|blocked', 'w'] });
    S('teambition', 'Teambition', 'pm', 'Teambition', 'teambition', { task: ['任务', 'task,tasks', CRUD], project: ['项目', 'project,projects', 'create list get'] });
    S('dropbox', 'Dropbox', 'office', 'Dropbox', 'dropbox', { file: ['文件', 'file,files', 'upload download delete move search share'], folder: ['文件夹', 'folder,folders', 'create list share'] });
    S('onedrive', 'OneDrive / SharePoint', 'office', 'OneDrive,SharePoint', 'onedrive,sharepoint', { file: ['文件', 'file,files', 'upload download delete move search share'], folder: ['文件夹', 'folder,folders', 'create list'], list: ['SharePoint 列表', 'SharePoint list,SharePoint lists', 'create get update'] });
    S('tencentdocs', '腾讯文档', 'office', '腾讯文档', 'tencent docs', { doc: ['在线文档,文档', 'doc,docs', 'create get update share'], sheet: ['在线表格,表格', 'sheet,sheets', 'create read write share'] }, { read: ['读表格', 'read sheet', 'Read cells of a {o}.', 'id:str range:str', 'r'], write: ['写表格', 'write sheet', 'Write cells of a {o}.', 'id:str range:str values:any', 'w'] });
    S('yuque', '语雀', 'office', '语雀', 'yuque', { doc: ['文档', 'doc,docs', 'create get update search'], repo: ['知识库', 'knowledge base,knowledge bases', 'create list'] });
    S('teams', 'Microsoft Teams', 'office', 'Teams,微软 Teams', 'teams,microsoft teams', { message: ['消息', 'message,messages', 'send reply list'], channel: ['频道', 'channel,channels', 'create list'], meeting: ['会议,在线会议', 'online meeting,online meetings', 'create list'] });
    S('discord', 'Discord', 'office', 'Discord', 'discord', { message: ['消息', 'message,messages', 'send reply delete list'], channel: ['频道', 'channel,channels', 'create list'], role: ['身份组,角色', 'role,roles', 'create assign'], thread: ['帖子,子区', 'thread,threads', 'create list'] });
    S('telegram', 'Telegram', 'office', 'Telegram,电报', 'telegram', { message: ['消息', 'message,messages', 'send reply'], chat: ['群,聊天', 'chat,chats', 'get list'] });
    S('dingtalk', '钉钉', 'office', '钉钉,DingTalk', 'dingtalk', { message: ['工作消息,消息', 'message,messages', 'send reply'], approval: ['审批', 'approval,approvals', 'create approve reject list'], attendance: ['考勤,打卡记录', 'attendance record,attendance records', 'list'], todo: ['待办', 'todo,todos', 'create list complete'] });
    S('wecom', '企业微信', 'office', '企业微信,企微,WeCom', 'wecom,wechat work', { message: ['应用消息,消息', 'message,messages', 'send'], customer: ['客户,外部联系人', 'external contact,external contacts', 'list get tag'], meeting: ['会议', 'meeting,meetings', 'create list'], approval: ['审批', 'approval,approvals', 'create list'] });
    S('zoom', 'Zoom', 'office', 'Zoom', 'zoom', { meeting: ['会议,视频会议', 'meeting,meetings', 'create get update delete list'], recording: ['会议录像,录制', 'recording,recordings', 'list download delete'] });
    S('tencentmeeting', '腾讯会议', 'office', '腾讯会议', 'tencent meeting,voov', { meeting: ['会议', 'meeting,meetings', 'create get delete list'], recording: ['录制,云录制', 'recording,recordings', 'list download'] });
    S('shopify', 'Shopify', 'commerce', 'Shopify,网店', 'shopify', { product: ['商品', 'product,products', CRUD], order: ['订单', 'order,orders', 'get list search cancel refund fulfill'], customer: ['顾客', 'customer,customers', 'create get update search'], inventory: ['库存', 'inventory level,inventory levels', 'get adjust'], discount: ['折扣,优惠码', 'discount code,discount codes', 'create list delete'] });
    S('stripe', 'Stripe', 'commerce', 'Stripe', 'stripe', { customer: ['客户', 'customer,customers', 'create get update list'], payment: ['付款,支付', 'payment intent,payment intents', 'create get list'], refund: ['退款单', 'refund,refunds', 'create list'], invoice: ['发票,账单', 'invoice,invoices', 'create get list send void'], subscription: ['订阅', 'subscription,subscriptions', 'create get update cancel list'], product: ['商品,价格', 'product,products', 'create list'] });
    S('paypal', 'PayPal', 'commerce', 'PayPal,贝宝', 'paypal', { order: ['订单', 'order,orders', 'create get capture'], refund: ['退款单', 'refund,refunds', 'create get'], payout: ['打款', 'payout,payouts', 'create get'] }, { capture: ['收款,确认扣款', 'capture payment', 'Capture an approved {o}.', 'id:str', 'o'] });
    S('mailchimp', 'Mailchimp', 'marketing', 'Mailchimp', 'mailchimp', { campaign: ['营销活动,群发邮件', 'campaign,campaigns', 'create get send schedule list'], audience: ['受众,订阅列表', 'audience,audiences', 'list get'], member: ['订阅者', 'subscriber,subscribers', 'create update delete search'] });
    S('sendgrid', 'SendGrid', 'marketing', 'SendGrid', 'sendgrid', { email: ['邮件', 'email,emails', 'send'], template: ['邮件模板', 'template,templates', 'create list'], stats: ['发送统计', 'email stats,email stats', 'get'] });
    S('googleads', 'Google Ads', 'marketing', 'Google Ads,谷歌广告', 'google ads,adwords', { campaign: ['广告系列,投放计划', 'campaign,campaigns', 'create get update pause resume list'], ad_group: ['广告组', 'ad group,ad groups', 'create update list'], keyword: ['关键词', 'keyword,keywords', 'create delete list'], report: ['投放报表,广告数据', 'performance report,performance reports', 'get'] });
    S('metaads', 'Meta Ads', 'marketing', 'Facebook 广告,Meta 广告', 'meta ads,facebook ads', { campaign: ['广告系列', 'campaign,campaigns', 'create get update pause resume list'], ad_set: ['广告组', 'ad set,ad sets', 'create update list'], creative: ['广告素材,创意', 'ad creative,ad creatives', 'create list'], insight: ['广告数据,投放效果', 'insight,insights', 'get'] });
    S('ga4', 'Google Analytics', 'data', 'Google Analytics,GA4,谷歌分析', 'google analytics,ga4', { report: ['报告,流量报表', 'report,reports', 'run'], realtime: ['实时数据,实时报告', 'realtime report,realtime reports', 'run'], property: ['媒体资源', 'property,properties', 'list get'] });
    S('mixpanel', 'Mixpanel', 'data', 'Mixpanel', 'mixpanel', { event: ['事件', 'event,events', 'query'], funnel: ['漏斗', 'funnel,funnels', 'query'], cohort: ['分群', 'cohort,cohorts', 'list'], report: ['报表', 'insights report,insights reports', 'run'] });
    S('amplitude', 'Amplitude', 'data', 'Amplitude', 'amplitude', { chart: ['图表', 'chart,charts', 'get create'], cohort: ['用户分群', 'cohort,cohorts', 'create list'], event: ['事件', 'event,events', 'query'] });
    S('tableau', 'Tableau', 'data', 'Tableau', 'tableau', { workbook: ['工作簿', 'workbook,workbooks', 'list get publish'], view: ['视图,图表', 'view,views', 'export_image export_csv'], datasource: ['数据源', 'data source,data sources', 'list refresh'] }, { export_image: ['导出图表图片', 'export view image', 'Export a {o} as PNG.', 'id:str path:str', 'r'], export_csv: ['导出数据', 'export view data', 'Export a {o} as CSV.', 'id:str path:str', 'r'], refresh: ['刷新数据源', 'refresh extract', 'Refresh a {o} extract.', 'id:str', 'w'] });
    S('powerbi', 'Power BI', 'data', 'Power BI,PowerBI', 'power bi', { report: ['报表', 'report,reports', 'list get export'], dataset: ['数据集', 'dataset,datasets', 'refresh list'], dashboard: ['仪表板', 'dashboard,dashboards', 'list get'] }, { refresh: ['刷新数据集', 'refresh dataset', 'Refresh a {o}.', 'id:str', 'w'] });
    S('gitlab', 'GitLab', 'dev', 'GitLab,极狐', 'gitlab', { merge_request: ['合并请求,MR', 'merge request,merge requests', 'create get list merge comment'], issue: ['议题,issue', 'issue,issues', CRUD], pipeline: ['流水线,CI', 'pipeline,pipelines', 'list get run cancel retry'], project: ['项目,仓库', 'project,projects', 'get list search'] }, { retry: ['重跑流水线', 'retry pipeline', 'Retry failed jobs of a {o}.', 'id:str', 'w'], cancel: ['取消流水线', 'cancel pipeline', 'Cancel a running {o}.', 'id:str', 'w'] });
    S('bitbucket', 'Bitbucket', 'dev', 'Bitbucket', 'bitbucket', { pull_request: ['PR,合并请求', 'pull request,pull requests', 'create list merge comment'], repository: ['仓库', 'repository,repositories', 'list get'] });
    S('jenkins', 'Jenkins', 'dev', 'Jenkins', 'jenkins', { job: ['任务,构建任务,Job', 'job,jobs', 'list get run'], build: ['构建,一次构建', 'build,builds', 'get logs cancel'], node: ['构建机,节点', 'build agent,build agents', 'list'] }, { cancel: ['取消构建,停掉构建', 'abort build', 'Abort a running {o}.', 'id:str', 'w'] });
    S('teamcity', 'TeamCity', 'dev', 'TeamCity', 'teamcity', { build: ['构建', 'build,builds', 'run get list logs cancel'], agent: ['构建代理', 'build agent,build agents', 'list'] }, { cancel: ['取消构建', 'cancel build', 'Cancel a queued or running {o}.', 'id:str', 'w'] });
    S('sonarqube', 'SonarQube', 'dev', 'SonarQube,代码扫描', 'sonarqube', { issue: ['代码问题,扫描结果', 'code issue,code issues', 'search'], project: ['项目', 'project,projects', 'get list'], quality_gate: ['质量门禁', 'quality gate,quality gates', 'get'] });
    S('vercel', 'Vercel', 'dev', 'Vercel', 'vercel', { deployment: ['部署', 'deployment,deployments', 'list get create cancel'], project: ['项目', 'project,projects', 'list get'], domain: ['域名', 'domain,domains', 'list add'] }, { cancel: ['取消部署', 'cancel deployment', 'Cancel a {o}.', 'id:str', 'w'] });
    S('cloudflare', 'Cloudflare', 'dev', 'Cloudflare,CF', 'cloudflare', { dns_record: ['DNS 记录,解析', 'DNS record,DNS records', CRUD], worker: ['Worker,边缘函数', 'worker,workers', 'list deploy logs'], cache: ['CDN 缓存', 'cache,caches', 'purge'], zone: ['站点,域名', 'zone,zones', 'list get'] }, { deploy: ['部署 Worker,上线', 'deploy worker', 'Deploy a {o} script.', 'name:str script:str', 'w'], purge: ['刷新 CDN,清缓存', 'purge cache', 'Purge {os} by URL or tag.', 'zone:str urls:str[]?', 'd'] });
    S('datadog', 'Datadog', 'dev', 'Datadog', 'datadog', { monitor: ['监控,告警规则', 'monitor,monitors', 'create get update list mute'], metric: ['指标', 'metric,metrics', 'query'], log: ['日志', 'log,logs', 'search'], incident: ['事故', 'incident,incidents', 'create list update'] }, { mute: ['静默告警', 'mute monitor', 'Mute a {o}.', 'id:str end:str?', 'w'] });
    S('pagerduty', 'PagerDuty', 'dev', 'PagerDuty,值班', 'pagerduty,on-call', { incident: ['告警事件,事故', 'incident,incidents', 'list get acknowledge resolve create'], oncall: ['值班表,谁在值班', 'on-call schedule,on-call schedules', 'get'] }, { acknowledge: ['确认告警,ack', 'acknowledge,ack', 'Acknowledge an {o}.', 'id:str', 'w'], resolve: ['解决事故,关闭告警', 'resolve incident', 'Resolve an {o}.', 'id:str', 'w'] });
    S('grafana', 'Grafana', 'dev', 'Grafana', 'grafana', { dashboard: ['看板,监控面板', 'dashboard,dashboards', 'search get create'], alert: ['告警', 'alert rule,alert rules', 'list create'], query: ['监控查询,PromQL', 'datasource query,datasource queries', 'run'] });
    S('azure', 'Azure', 'dev', 'Azure,微软云', 'azure', { vm: ['虚拟机', 'virtual machine,virtual machines', 'list start stop restart'], storage: ['存储,Blob', 'blob,blobs', 'list upload download delete'], function: ['Azure Functions,函数', 'function app,function apps', 'list deploy logs'] }, { deploy: ['部署,发布,上线', 'deploy,publish', 'Deploy a {o}.', 'name:str zip_path:str', 'w'] });
    S('gcp', 'Google Cloud', 'dev', 'GCP,谷歌云', 'gcp,google cloud', { instance: ['云主机,虚拟机', 'compute instance,compute instances', 'list start stop'], bucket: ['存储桶,GCS', 'storage bucket,storage buckets', 'list create'], object: ['GCS 文件', 'storage object,storage objects', 'upload download delete list'], bigquery: ['BigQuery 查询,数仓查询', 'BigQuery query,BigQuery queries', 'run'] });
    S('aliyun', '阿里云', 'dev', '阿里云,Aliyun', 'aliyun,alibaba cloud', { ecs: ['ECS,云服务器', 'ECS instance,ECS instances', 'list start stop restart'], oss: ['OSS 文件,对象存储', 'OSS object,OSS objects', 'upload download delete list'], sls: ['日志服务,SLS 日志', 'log store query,log store queries', 'run'], cdn: ['CDN', 'CDN cache,CDN caches', 'refresh'] }, { refresh: ['刷新 CDN,刷新缓存', 'refresh cdn', 'Refresh {os} for URLs.', 'urls:str[]', 'd'] });
    S('tencentcloud', '腾讯云', 'dev', '腾讯云', 'tencent cloud', { cvm: ['CVM,云服务器', 'CVM instance,CVM instances', 'list start stop restart'], cos: ['COS 文件,对象存储', 'COS object,COS objects', 'upload download delete list'], cls: ['日志服务,CLS', 'CLS log,CLS logs', 'search'] });
    S('steam', 'Steamworks', 'gameops', 'Steam,Steamworks', 'steam,steamworks', { build: ['构建,游戏包', 'build,builds', 'upload set_live list'], achievement: ['成就', 'achievement,achievements', 'create update list'], leaderboard: ['排行榜', 'leaderboard,leaderboards', 'create get'], news: ['公告,新闻', 'announcement,announcements', 'create'], key: ['激活码,CDKey', 'product key,product keys', 'create'] }, { set_live: ['设为正式分支,上线这个版本', 'set build live,publish branch', 'Set a {o} live on a branch.', 'build_id:str branch:str', 'o'], upload: ['上传游戏包,传包', 'upload build', 'Upload a {o} with SteamPipe.', 'depot:str path:str', 'o'] });
    S('appstore', 'App Store Connect', 'gameops', 'App Store,苹果后台,TestFlight', 'app store connect,testflight', { build: ['构建,iOS 包', 'build,builds', 'list get submit'], tester: ['测试员', 'beta tester,beta testers', 'invite list'], review: ['用户评价,评论', 'customer review,customer reviews', 'list reply'], version: ['版本', 'app version,app versions', 'create submit'] }, { submit: ['提交审核,送审', 'submit for review', 'Submit a {o} for review.', 'id:str', 'o'] });
    S('googleplay', 'Google Play Console', 'gameops', 'Google Play,谷歌商店', 'google play,play console', { release: ['发布,版本', 'release,releases', 'create get promote'], review: ['评价', 'review,reviews', 'list reply'], vitals: ['崩溃率,ANR', 'android vital,android vitals', 'get'] }, { promote: ['推到正式,全量', 'promote release', 'Promote a {o} to another track.', 'id:str track:internal|alpha|beta|production', 'o'] });
    S('taptap', 'TapTap 开发者服务', 'gameops', 'TapTap,Tap', 'taptap', { app: ['游戏页,商店页', 'game listing,game listings', 'get update'], review: ['评价,玩家评价', 'review,reviews', 'list reply'], test: ['测试资格,测试招募', 'test recruitment,test recruitments', 'create list'] });
    S('playfab', 'PlayFab', 'gameops', 'PlayFab', 'playfab', { player: ['玩家', 'player,players', 'get ban'], catalog: ['商品目录,道具', 'catalog item,catalog items', 'create update list'], title_data: ['配置,Title Data', 'title data entry,title data entries', 'get set'], cloudscript: ['云代码,CloudScript', 'cloudscript function,cloudscript functions', 'run'] }, { ban: ['封号,封禁玩家', 'ban player', 'Ban a {o}.', 'id:str reason:str hours:int?', 'd'] });
    S('firebase', 'Firebase', 'gameops', 'Firebase', 'firebase', { crash: ['崩溃,Crashlytics', 'crash issue,crash issues', 'list get'], remote_config: ['远程配置', 'remote config parameter,remote config parameters', 'get set publish'], message: ['推送,FCM', 'push notification,push notifications', 'send'], user: ['用户', 'auth user,auth users', 'get delete'] });
    S('bugly', 'Bugly', 'gameops', 'Bugly,腾讯 Bugly', 'bugly', { crash: ['崩溃,闪退', 'crash issue,crash issues', 'list get'], anr: ['ANR,卡死', 'ANR issue,ANR issues', 'list'], version: ['版本', 'app version,app versions', 'list'] });
    S('crowdin', 'Crowdin', 'gameops', 'Crowdin,本地化平台,翻译平台', 'crowdin,localization', { file: ['源文件,待翻译文件', 'source file,source files', 'upload list'], translation: ['翻译,译文', 'translation,translations', 'export build'], string: ['词条,文本', 'string,strings', 'search update'], glossary: ['术语表', 'glossary term,glossary terms', 'create list'] });
    S('canva', 'Canva', 'dcc', 'Canva,可画', 'canva', { design: ['设计,海报', 'design,designs', 'create list export'], asset: ['素材', 'asset,assets', 'upload list'] });
    S('miro', 'Miro', 'office', 'Miro,白板', 'miro,whiteboard', { board: ['白板', 'board,boards', 'create list get'], sticky_note: ['便签', 'sticky note,sticky notes', 'create list'] });
    S('airtable', 'Airtable', 'office', 'Airtable', 'airtable', { record: ['记录,行', 'record,records', CRUD], table: ['表', 'table,tables', 'list get'] });
    S('typeform', 'Typeform / 问卷', 'office', 'Typeform,问卷星', 'typeform,survey', { form: ['问卷,表单', 'form,forms', 'create list get'], response: ['答卷,回收结果', 'response,responses', 'list export'] });
    S('quickbooks', 'QuickBooks', 'finance', 'QuickBooks', 'quickbooks', { invoice: ['发票', 'invoice,invoices', 'create get list send'], customer: ['客户', 'customer,customers', 'create list'], expense: ['费用,报销', 'expense,expenses', 'create list'] });
    S('workday', 'Workday / HR', 'finance', 'Workday,人事系统', 'workday,hr', { employee: ['员工', 'employee,employees', 'get search'], time_off: ['请假', 'time off request,time off requests', 'create list approve'] });

    // —— 云平台与大应用的「深度」：真实目录里成千上万个工具，大头来自这里 ——
    // X(ns, 对象, 自定义动词?)：给已有应用追加对象
    const X = (ns, objects, verbs, exact) => { const a = APPS.find((q) => q.ns === ns); for (const k of Object.keys(verbs || {})) if (a.verbs[k]) throw new Error(`verb ${ns}.${k} already defined`); Object.assign(a.objects, objects); Object.assign(a.verbs, verbs || {}); if (exact) a.exact.push(...exact); };
    X('aws', {
        bucket: ['S3 桶,存储桶', 'S3 bucket,S3 buckets', 'list create delete get_policy set_policy'],
        object: ['S3 文件,对象存储文件', 'S3 object,S3 objects', 'list upload download delete copy presign'],
        instance: ['EC2,云服务器,实例', 'EC2 instance,EC2 instances', 'list start stop reboot terminate get create'],
        security_group: ['安全组', 'security group,security groups', 'list create authorize_ingress'],
        volume: ['EBS 卷,云硬盘', 'EBS volume,EBS volumes', 'list create attach snapshot'],
        function: ['Lambda,云函数', 'Lambda function,Lambda functions', 'list invoke create update_code update_config delete logs'],
        log: ['CloudWatch 日志,云日志', 'CloudWatch log,CloudWatch logs', 'query tail'],
        metric: ['CloudWatch 指标,监控指标', 'CloudWatch metric,CloudWatch metrics', 'get put'],
        alarm: ['CloudWatch 告警', 'CloudWatch alarm,CloudWatch alarms', 'create list delete'],
        iam_user: ['IAM 用户', 'IAM user,IAM users', 'list create delete'],
        iam_role: ['IAM 角色', 'IAM role,IAM roles', 'list create attach_policy'],
        access_key: ['访问密钥,AK', 'access key,access keys', 'create rotate delete'],
        dynamodb_table: ['DynamoDB 表', 'DynamoDB table,DynamoDB tables', 'list create get delete'],
        dynamodb_item: ['DynamoDB 数据', 'DynamoDB item,DynamoDB items', 'get put query scan delete'],
        rds_instance: ['RDS 数据库实例', 'RDS instance,RDS instances', 'list start stop snapshot restore get'],
        ecs_service: ['ECS 服务', 'ECS service,ECS services', 'list update scale restart'],
        ecs_task: ['ECS 任务', 'ECS task,ECS tasks', 'list run stop logs'],
        eks_cluster: ['EKS 集群', 'EKS cluster,EKS clusters', 'list get update_kubeconfig'],
        sqs_queue: ['SQS 队列,消息队列', 'SQS queue,SQS queues', 'list create send receive purge delete'],
        sns_topic: ['SNS 主题,通知主题', 'SNS topic,SNS topics', 'list create publish subscribe'],
        dns_record: ['Route 53 记录,域名解析', 'Route 53 record,Route 53 records', 'list create update delete'],
        distribution: ['CloudFront 分发,CDN', 'CloudFront distribution,CloudFront distributions', 'list get invalidate'],
        secret: ['Secrets Manager 密钥,密钥', 'secret,secrets', 'get create update rotate list'],
        parameter: ['SSM 参数,参数存储', 'SSM parameter,SSM parameters', 'get put list'],
        ecr_repository: ['ECR 镜像仓库', 'ECR repository,ECR repositories', 'list create'],
        stack: ['CloudFormation 栈,基础设施栈', 'CloudFormation stack,CloudFormation stacks', 'list create update delete get events'],
        codebuild_project: ['CodeBuild 项目', 'CodeBuild project,CodeBuild projects', 'list start_build'],
        athena_query: ['Athena 查询', 'Athena query,Athena queries', 'run get_results'],
        cost: ['账单,费用,AWS 花了多少钱', 'cost report,cost reports', 'get forecast'],
    }, {
        get_policy: ['看桶策略', 'get bucket policy', 'Get the policy of an {o}.', 'bucket:str', 'r'], set_policy: ['改桶策略,设置权限策略', 'set bucket policy', 'Set the policy of an {o}.', 'bucket:str policy:obj', 'd'],
        copy: ['复制对象', 'copy object', 'Copy an {o}.', 'source:str destination:str', 'w'], presign: ['生成下载链接,预签名', 'presigned url', 'Create a presigned URL for an {o}.', 'bucket:str key:str expires_s:int?', 'r'],
        reboot: ['重启服务器', 'reboot instance', 'Reboot an {o}.', 'id:str', 'w'], terminate: ['销毁实例,释放服务器', 'terminate instance', 'Terminate an {o}. This cannot be undone.', 'id:str', 'd'],
        authorize_ingress: ['开放端口,加入站规则', 'open port,authorize ingress', 'Add an inbound rule to a {o}.', 'group_id:str port:int cidr:str', 'd'],
        attach: ['挂载硬盘', 'attach volume', 'Attach an {o} to an instance.', 'volume_id:str instance_id:str', 'w'], snapshot: ['打快照,备份', 'snapshot,backup', 'Create a snapshot of an {o}.', 'id:str', 'w'],
        invoke: ['调用云函数', 'invoke lambda', 'Invoke a {o}.', 'name:str payload:obj?', 'o'], update_code: ['更新函数代码,部署 Lambda', 'deploy lambda,update function code', 'Update the code of a {o}.', 'name:str zip_path:str', 'w'], update_config: ['改函数配置,改内存超时', 'update function configuration', 'Update memory, timeout or env of a {o}.', 'name:str memory_mb:int? timeout_s:int? env:obj?', 'w'],
        tail: ['实时看日志,跟踪日志', 'tail logs', 'Tail {os} live.', 'group:str filter:str?', 'r'], put: ['写入', 'put', 'Put a {o}.', 'name:str value:any', 'w'],
        attach_policy: ['授权,挂策略', 'attach policy', 'Attach a managed policy to an {o}.', 'role:str policy_arn:str', 'd'], rotate: ['轮换密钥', 'rotate', 'Rotate an {o}.', 'id:str', 'd'],
        scan: ['全表扫描', 'scan table', 'Scan {os} (expensive).', 'table:str filter:str?', 'r'], restore: ['从快照恢复', 'restore from snapshot', 'Restore an {o} from a snapshot.', 'snapshot_id:str', 'w'],
        scale: ['扩容,缩容,改副本数', 'scale service', 'Set the desired count of an {o}.', 'service:str count:int', 'w'], update_kubeconfig: ['配置 kubectl,拿 kubeconfig', 'update kubeconfig', 'Write kubeconfig for an {o}.', 'name:str', 'w'],
        receive: ['收消息,拉消息', 'receive messages', 'Receive messages from an {o}.', 'queue_url:str max:int?', 'r'], purge: ['清空队列', 'purge queue', 'Delete all messages in an {o}.', 'queue_url:str', 'd'],
        subscribe: ['订阅主题', 'subscribe', 'Subscribe an endpoint to an {o}.', 'topic_arn:str protocol:email|sqs|lambda|https endpoint:str', 'w'],
        invalidate: ['刷新 CDN,清 CloudFront 缓存', 'invalidate cache,create invalidation', 'Create an invalidation for a {o}.', 'id:str paths:str[]', 'd'],
        events: ['看栈事件,部署进度', 'stack events', 'List events of a {o}.', 'name:str', 'r'], start_build: ['触发构建', 'start build', 'Start a build of a {o}.', 'name:str', 'w'],
        get_results: ['查询结果', 'query results', 'Get the results of an {o}.', 'id:str', 'r'], forecast: ['费用预测', 'cost forecast', 'Forecast {os}.', 'start_date:str end_date:str', 'r'],
    });
    X('gcp', {
        gke_cluster: ['GKE 集群', 'GKE cluster,GKE clusters', 'list get get_credentials'],
        cloud_run: ['Cloud Run 服务', 'Cloud Run service,Cloud Run services', 'list deploy logs'],
        pubsub_topic: ['Pub/Sub 主题', 'Pub/Sub topic,Pub/Sub topics', 'list create publish'],
        cloud_sql: ['Cloud SQL 实例', 'Cloud SQL instance,Cloud SQL instances', 'list start stop backup'],
        service_account: ['服务账号', 'service account,service accounts', 'list create create_key'],
        secret: ['Secret Manager 密钥', 'secret,secrets', 'get create list'],
        log: ['Cloud Logging 日志', 'log entry,log entries', 'query'],
    }, { get_credentials: ['拿集群凭证', 'get credentials', 'Fetch kubeconfig credentials of a {o}.', 'name:str region:str', 'w'], deploy: ['部署服务', 'deploy service', 'Deploy a revision of a {o}.', 'name:str image:str region:str', 'w'], backup: ['备份数据库', 'backup', 'Create a backup of a {o}.', 'name:str', 'w'], create_key: ['生成密钥文件', 'create key', 'Create a key for a {o}.', 'email:str', 'd'] });
    X('azure', {
        aks_cluster: ['AKS 集群', 'AKS cluster,AKS clusters', 'list get get_credentials'],
        app_service: ['App Service,Web 应用', 'web app,web apps', 'list restart deploy logs'],
        sql_db: ['Azure SQL 数据库', 'Azure SQL database,Azure SQL databases', 'list get'],
        key_vault: ['Key Vault 密钥', 'key vault secret,key vault secrets', 'get set list'],
        devops_pipeline: ['Azure DevOps 流水线', 'pipeline,pipelines', 'list run get'],
        monitor_alert: ['Azure 监控告警', 'alert rule,alert rules', 'list create'],
    }, { get_credentials: ['拿 AKS 凭证', 'get aks credentials', 'Fetch kubeconfig of an {o}.', 'name:str resource_group:str', 'w'] });
    X('aliyun', {
        rds: ['RDS,云数据库', 'RDS instance,RDS instances', 'list get backup restart'],
        ack: ['ACK 集群,容器服务', 'ACK cluster,ACK clusters', 'list get'],
        fc: ['函数计算,FC', 'FC function,FC functions', 'list invoke deploy logs'],
        ram_user: ['RAM 用户,子账号', 'RAM user,RAM users', 'list create delete'],
        slb: ['负载均衡,SLB', 'SLB instance,SLB instances', 'list get set_backend'],
        sms: ['短信,短信服务', 'SMS message,SMS messages', 'send query'],
        cms_alarm: ['云监控告警', 'cloud monitor alarm,cloud monitor alarms', 'list create'],
    }, { backup: ['备份数据库', 'backup database', 'Back up an {o}.', 'id:str', 'w'], invoke: ['调用函数', 'invoke function', 'Invoke an {o}.', 'name:str payload:obj?', 'o'], deploy: ['部署函数', 'deploy function', 'Deploy code to an {o}.', 'name:str zip_path:str', 'w'], set_backend: ['改后端服务器,挂服务器', 'set backend servers', 'Set backend servers of an {o}.', 'id:str servers:str[]', 'w'], query: ['查发送记录', 'query sms', 'Query delivery records of {os}.', 'phone:str date:str', 'r'] });
    X('tencentcloud', {
        tke: ['TKE 集群,容器集群', 'TKE cluster,TKE clusters', 'list get'],
        scf: ['云函数,SCF', 'SCF function,SCF functions', 'list invoke deploy logs'],
        cdb: ['云数据库 MySQL,CDB', 'CDB instance,CDB instances', 'list get backup restart'],
        cdn: ['CDN', 'CDN cache,CDN caches', 'refresh'],
        vod: ['云点播,视频点播', 'VOD media,VOD media', 'upload list transcode'],
        sms: ['短信', 'SMS message,SMS messages', 'send'],
        gme: ['GME 语音,游戏语音', 'GME room,GME rooms', 'list kick'],
    }, { invoke: ['调用云函数', 'invoke scf', 'Invoke an {o}.', 'name:str payload:obj?', 'o'], deploy: ['部署云函数', 'deploy scf', 'Deploy an {o}.', 'name:str zip_path:str', 'w'], backup: ['备份', 'backup', 'Back up an {o}.', 'id:str', 'w'], refresh: ['刷新 CDN', 'purge cdn', 'Refresh {os} for URLs.', 'urls:str[]', 'd'], transcode: ['转码', 'transcode', 'Transcode {o}.', 'file_id:str template:str', 'w'], kick: ['踢出房间', 'kick user', 'Remove a user from a {o}.', 'room:str user:str', 'd'] });
    X('unreal', {
        data_table: ['数据表,DataTable,配置表', 'data table,data tables', 'create import_csv get_row set_row'],
        data_asset: ['数据资产,DataAsset', 'data asset,data assets', 'create set_property'],
        gameplay_tag: ['GameplayTag,游戏标签', 'gameplay tag,gameplay tags', 'add list'],
        input_action: ['输入动作,Enhanced Input', 'input action,input actions', 'create add_mapping'],
        anim_blueprint: ['动画蓝图', 'animation blueprint,animation blueprints', 'create compile'],
        montage: ['动画蒙太奇', 'anim montage,anim montages', 'create'],
        texture: ['贴图,纹理', 'texture,textures', 'import set_compression set_lod_group'],
        sound: ['声音,音效', 'sound wave,sound waves', 'import assign_attenuation'],
        landscape: ['地形,Landscape', 'landscape,landscapes', 'create import_heightmap add_layer'],
        foliage: ['植被', 'foliage type,foliage types', 'create paint'],
        world_partition: ['World Partition,大世界', 'world partition region,world partition regions', 'load build_hlod'],
        pcg_graph: ['PCG,程序化生成', 'PCG graph,PCG graphs', 'create generate'],
        localization: ['本地化,多语言文本', 'localization target,localization targets', 'gather compile'],
        plugin: ['插件', 'plugin,plugins', 'list enable disable'],
        editor_utility: ['编辑器工具,Editor Utility', 'editor utility widget,editor utility widgets', 'run'],
    }, {
        import_csv: ['从 CSV 导入,导表', 'import csv', 'Fill a {o} from a CSV file.', 'table:str path:str', 'w'], get_row: ['读一行配置', 'get row', 'Read one row of a {o}.', 'table:str row:str', 'r'], set_row: ['改一行配置', 'set row', 'Write one row of a {o}.', 'table:str row:str values:obj', 'w'],
        add_mapping: ['绑定按键,加键位映射', 'add key mapping', 'Map a key to an {o} in an Input Mapping Context.', 'context:str action:str key:str', 'w'],
        set_compression: ['贴图压缩,压缩格式', 'texture compression', 'Set compression settings of a {o}.', 'texture:str setting:Default|NormalMap|Masks|UserInterface2D|HDR', 'w'], set_lod_group: ['贴图 LOD 组', 'texture group', 'Set the LOD group of a {o}.', 'texture:str group:str', 'w'],
        assign_attenuation: ['声音衰减,设置衰减', 'sound attenuation', 'Assign attenuation settings to a {o}.', 'sound:str attenuation:str', 'w'],
        import_heightmap: ['导入高度图', 'import heightmap', 'Import a heightmap into a {o}.', 'path:str', 'w'], add_layer: ['加地形图层,加材质层', 'add paint layer', 'Add a paint layer to a {o}.', 'name:str', 'w'],
        paint: ['刷植被,种树,种草', 'paint foliage,scatter foliage', 'Paint / scatter {os} in an area.', 'type:str center:vec3 radius:num density:num?', 'w'],
        load: ['加载区域', 'load region', 'Load a {o} in the editor.', 'min:vec3 max:vec3', 'w'], build_hlod: ['构建 HLOD', 'build hlod', 'Build HLODs for {os}.', '', 'w'],
        generate: ['生成 PCG', 'generate pcg', 'Generate a {o} in the level.', 'graph:str seed:int?', 'w'],
        gather: ['收集文本', 'gather text', 'Gather text for a {o}.', 'target:str', 'w'],
        enable: ['启用插件', 'enable plugin', 'Enable a {o}.', 'name:str', 'w'], disable: ['禁用插件', 'disable plugin', 'Disable a {o}.', 'name:str', 'w'],
    });
    X('unity', {
        addressable: ['Addressables,可寻址资源', 'addressable group,addressable groups', 'create mark build_bundles'],
        scriptable_object: ['ScriptableObject,配置资源', 'ScriptableObject,ScriptableObjects', 'create set_field'],
        timeline: ['Timeline,过场', 'timeline,timelines', 'create add_track'],
        navmesh: ['导航网格,NavMesh', 'NavMesh,NavMeshes', 'build'],
        terrain: ['地形', 'terrain,terrains', 'create set_height paint_texture'],
        particle_system: ['粒子系统,特效', 'particle system,particle systems', 'create set_property'],
        input_action: ['输入动作,Input System', 'input action,input actions', 'create bind'],
        shader_graph: ['Shader Graph', 'shader graph,shader graphs', 'create'],
        quality: ['画质等级,质量设置', 'quality level,quality levels', 'set'],
        cinemachine: ['Cinemachine,虚拟相机', 'virtual camera,virtual cameras', 'create set_follow'],
    }, { mark: ['标成可寻址', 'mark addressable', 'Mark assets as {os}.', 'paths:str[] group:str', 'w'], build_bundles: ['构建 Addressables,打 AB 包', 'build addressables,build bundles', 'Build {os} content bundles.', '', 'w'], set_field: ['改字段', 'set field', 'Set a field of a {o}.', 'path:str field:str value:any', 'w'], add_track: ['加轨道', 'add track', 'Add a track to a {o}.', 'path:str type:Animation|Activation|Audio|Signal', 'w'], set_height: ['改地形高度', 'set terrain height', 'Set heights of a {o} from a heightmap.', 'terrain:str heightmap:str', 'w'], paint_texture: ['刷地形材质', 'paint terrain texture', 'Paint a texture layer on a {o}.', 'terrain:str layer:str', 'w'], bind: ['绑定按键', 'bind key', 'Bind a key to an {o}.', 'action:str binding:str', 'w'], set_follow: ['跟随目标', 'set follow target', 'Set Follow / LookAt of a {o}.', 'camera:str target:str', 'w'] });
    X('blender', {
        geometry_nodes: ['几何节点,GN', 'geometry node tree,geometry node trees', 'create add_node connect'],
        shader_nodes: ['着色器节点,材质节点', 'shader node,shader nodes', 'add connect'],
        armature: ['骨架,骨骼', 'armature,armatures', 'create add_bone auto_weight'],
        collection: ['集合', 'collection,collections', 'create move_to'],
        world: ['环境光,世界,HDRI', 'world,worlds', 'set_hdri'],
        light: ['灯光,光源', 'light,lights', 'create set_energy'],
        camera: ['相机', 'camera,cameras', 'create set_lens'],
        texture_bake: ['烘焙贴图,Bake', 'texture bake,texture bakes', 'run'],
    }, { add_node: ['加节点', 'add node', 'Add a node to a {o}.', 'tree:str type:str', 'w'], connect: ['连节点', 'connect nodes', 'Link two sockets in a {o}.', 'tree:str from:str to:str', 'w'], add_bone: ['加骨头', 'add bone', 'Add a bone to an {o}.', 'armature:str name:str head:vec3 tail:vec3', 'w'], auto_weight: ['自动权重,自动蒙皮', 'automatic weights', 'Parent a mesh to an {o} with automatic weights.', 'mesh:str armature:str', 'w'], move_to: ['移到集合', 'move to collection', 'Move objects to a {o}.', 'objects:str[] collection:str', 'w'], set_hdri: ['换 HDRI,换环境贴图', 'set hdri', 'Set an HDRI environment on the {o}.', 'path:str strength:num?', 'w'], set_energy: ['灯光强度,功率', 'light power,energy', 'Set the power of a {o} in watts.', 'light:str energy:num', 'w'], set_lens: ['焦距,镜头', 'focal length,lens', 'Set the focal length of a {o}.', 'camera:str lens_mm:num', 'w'] });
    X('photoshop', {
        smart_object: ['智能对象', 'smart object,smart objects', 'convert replace_contents'],
        layer_style: ['图层样式,描边,投影', 'layer style,layer styles', 'add_stroke add_drop_shadow'],
        artboard: ['画板', 'artboard,artboards', 'create export'],
        color: ['颜色模式,色彩配置', 'color profile,color profiles', 'convert'],
    }, { replace_contents: ['替换内容', 'replace contents', 'Replace contents of a {o}.', 'layer:str path:str', 'w'], add_stroke: ['加描边', 'add stroke', 'Add a stroke {o}.', 'layer:str size:int color:vec3', 'w'], add_drop_shadow: ['加投影', 'add drop shadow', 'Add a drop shadow {o}.', 'layer:str distance:int? size:int?', 'w'] }, [['generative_fill', 'AI 生成,生成式填充,AI 扩图', 'generative fill,ai fill', 'Run Generative Fill inside the current selection with a text prompt.', 'prompt:str', 'o']]);
    X('excel', {
        table: ['表格对象,超级表', 'table,tables', 'create'],
        validation: ['数据验证,下拉列表', 'data validation rule,data validation rules', 'add'],
        protection: ['保护,锁定工作表', 'sheet protection,sheet protections', 'protect unprotect'],
        name: ['名称,命名区域', 'defined name,defined names', 'create list'],
        pane: ['冻结窗格', 'frozen pane,frozen panes', 'freeze'],
        csv: ['CSV', 'CSV file,CSV files', 'import export'],
    }, { protect: ['保护工作表,加密码', 'protect sheet', 'Protect a sheet.', 'sheet:str password:str?', 'w'], unprotect: ['取消保护', 'unprotect sheet', 'Unprotect a sheet.', 'sheet:str password:str?', 'w'], freeze: ['冻结首行,冻结窗格', 'freeze panes', 'Freeze rows / columns.', 'sheet:str rows:int columns:int?', 'w'] });
    S('okta', 'Okta', 'dev', 'Okta,统一身份', 'okta,sso', { user: ['用户,账号', 'user,users', 'create get update deactivate list'], group: ['用户组', 'group,groups', 'list add_member'] }, { deactivate: ['停用账号', 'deactivate user', 'Deactivate a {o}.', 'id:str', 'd'], add_member: ['加人进组', 'add member', 'Add a user to a {o}.', 'group:str user:str', 'w'] });
    S('twilio', 'Twilio', 'office', 'Twilio', 'twilio', { sms: ['短信', 'SMS message,SMS messages', 'send list'], call: ['电话', 'phone call,phone calls', 'create list'] });
    S('snowflake', 'Snowflake', 'data', 'Snowflake,数仓', 'snowflake,warehouse', { query: ['SQL 查询,数仓查询', 'query,queries', 'run'], warehouse: ['计算仓库', 'warehouse,warehouses', 'list resume suspend'], table: ['表', 'table,tables', 'list get'] }, { suspend: ['挂起仓库', 'suspend warehouse', 'Suspend a {o}.', 'name:str', 'w'] });
    S('databricks', 'Databricks', 'data', 'Databricks', 'databricks', { job: ['作业,Job', 'job,jobs', 'list run get'], cluster: ['集群', 'cluster,clusters', 'list start stop'], notebook: ['Notebook,笔记本', 'notebook,notebooks', 'run export'] });
    S('airflow', 'Apache Airflow', 'data', 'Airflow,调度任务', 'airflow', { dag: ['DAG,工作流', 'DAG,DAGs', 'list trigger pause'], dag_run: ['运行记录', 'DAG run,DAG runs', 'list get'], task: ['任务', 'task instance,task instances', 'logs clear'] }, { trigger: ['触发工作流,手动跑一次', 'trigger dag', 'Trigger a {o}.', 'dag_id:str conf:obj?', 'w'], clear: ['重跑任务,清除状态', 'clear task', 'Clear {os} so they re-run.', 'dag_id:str task_id:str', 'w'] });
    S('kafka', 'Kafka', 'dev', 'Kafka,消息队列', 'kafka', { topic: ['Topic,主题', 'topic,topics', 'list create delete get'], consumer_group: ['消费组', 'consumer group,consumer groups', 'list get reset_offsets'], message: ['消息', 'message,messages', 'produce consume'] }, { reset_offsets: ['重置位点', 'reset offsets', 'Reset offsets of a {o}.', 'group:str topic:str to:earliest|latest', 'd'], produce: ['发消息到 Kafka', 'produce message', 'Produce a {o}.', 'topic:str value:str key:str?', 'w'], consume: ['消费消息', 'consume messages', 'Consume {os}.', 'topic:str max:int?', 'r'] });
    S('elasticsearch', 'Elasticsearch', 'dev', 'Elasticsearch,ES,检索', 'elasticsearch,es', { index: ['索引', 'index,indices', 'list create delete get_mapping'], document: ['文档', 'document,documents', 'search get index_doc delete'] }, { get_mapping: ['看 mapping', 'get mapping', 'Get the mapping of an {o}.', 'index:str', 'r'], index_doc: ['写入文档', 'index document', 'Index a {o}.', 'index:str body:obj id:str?', 'w'] });
    S('mongodb', 'MongoDB', 'dev', 'MongoDB,Mongo', 'mongodb,mongo', { collection: ['集合', 'collection,collections', 'list create drop'], document: ['文档', 'document,documents', 'find insert update delete aggregate'] }, { drop: ['删集合', 'drop collection', 'Drop a {o}.', 'name:str', 'd'], find: ['查文档', 'find documents', 'Find {os}.', 'collection:str filter:obj limit:int?', 'r'], insert: ['插入文档', 'insert document', 'Insert {os}.', 'collection:str docs:obj', 'w'], aggregate: ['聚合查询', 'aggregate', 'Run an aggregation pipeline.', 'collection:str pipeline:obj', 'r'] });
    S('supabase', 'Supabase', 'dev', 'Supabase', 'supabase', { table: ['表', 'table,tables', 'list query'], function: ['边缘函数', 'edge function,edge functions', 'list deploy'], migration: ['数据库迁移', 'migration,migrations', 'list apply'] }, { deploy: ['部署边缘函数', 'deploy edge function', 'Deploy an {o}.', 'name:str', 'w'] });
    S('postman', 'Postman', 'dev', 'Postman', 'postman', { collection: ['接口集合', 'collection,collections', 'list get run'], environment: ['环境变量', 'environment,environments', 'list update'] });
    S('statuspage', 'Statuspage', 'dev', 'Statuspage,状态页', 'statuspage,status page', { incident: ['故障公告', 'incident,incidents', 'create update resolve'], component: ['组件状态', 'component,components', 'list update'] }, { resolve: ['关闭故障公告', 'resolve incident', 'Resolve an {o}.', 'id:str', 'o'] });
    S('docusign', 'DocuSign', 'office', 'DocuSign,电子签', 'docusign,e-signature', { envelope: ['签署包,合同', 'envelope,envelopes', 'create send get void'] });
    S('calendly', 'Calendly', 'office', 'Calendly', 'calendly', { event_type: ['预约类型', 'event type,event types', 'list'], scheduled_event: ['预约', 'scheduled event,scheduled events', 'list cancel'] });
    S('ugs', 'Unity Gaming Services', 'gameops', 'UGS,Unity 云服务', 'unity gaming services,ugs', { cloud_save: ['云存档', 'cloud save item,cloud save items', 'get set delete'], economy: ['经济系统,虚拟货币', 'economy currency,economy currencies', 'list create'], remote_config: ['远程配置', 'remote config setting,remote config settings', 'get set'], leaderboard: ['排行榜', 'leaderboard,leaderboards', 'create get'] });
    S('eos', 'Epic Online Services', 'gameops', 'EOS,Epic 在线服务', 'epic online services,eos', { player: ['玩家', 'player,players', 'get'], achievement: ['成就', 'achievement,achievements', 'list create'], stat: ['统计', 'stat,stats', 'get'], sanction: ['处罚,封禁', 'sanction,sanctions', 'create list'] });
    S('agora', '声网 Agora', 'gameops', '声网,Agora', 'agora', { channel: ['频道,语音房', 'channel,channels', 'list kick'], recording: ['云录制', 'cloud recording,cloud recordings', 'start stop'] }, { kick: ['踢人', 'kick user', 'Kick a user from a {o}.', 'channel:str uid:str', 'd'] });
    S('photon', 'Photon', 'gameops', 'Photon,光子引擎', 'photon,photon engine', { room: ['房间', 'room,rooms', 'list get'], webhook: ['Webhook', 'webhook,webhooks', 'list update'] });
    S('wechatmp', '微信公众号', 'marketing', '公众号,微信公众号', 'wechat official account', { article: ['图文,文章', 'article,articles', 'create publish list'], user: ['粉丝', 'follower,followers', 'list get tag'], menu: ['自定义菜单', 'menu,menus', 'get update'] });
    S('bilibili', 'B 站开放平台', 'marketing', 'B站,哔哩哔哩,bilibili', 'bilibili', { video: ['投稿,视频', 'video,videos', 'upload list get'], comment: ['评论', 'comment,comments', 'list reply'], stat: ['数据,播放量', 'video stat,video stats', 'get'] });
    S('douyin', '抖音开放平台', 'marketing', '抖音', 'douyin', { video: ['视频,作品', 'video,videos', 'upload list get'], comment: ['评论', 'comment,comments', 'list reply'], data: ['数据,播放数据', 'video data,video data', 'get'] });

    // —— 没有接口时：操作屏幕 ——
    A('computer', '电脑操作（Computer Use）', 'gui', '屏幕,鼠标,键盘,界面操作,点界面', 'screen,mouse,keyboard,gui,computer use', '截图 + 鼠标键盘（最后的手段）', 'Operate any desktop application through screenshots, mouse and keyboard. Slow; use only when no API exists.', {}, {}, [
        ['screenshot', '截屏,看看屏幕', 'take screenshot,look at the screen', 'Capture the screen (about 1,560 image tokens at 1456×819).', '', 'r'],
        ['left_click', '点击屏幕,点这个按钮', 'click at,click on screen', 'Click at pixel coordinates.', 'x:int y:int', 'w'],
        ['type_text', '键盘输入,打字', 'type text,keyboard input', 'Type a string with the keyboard.', 'text:str', 'w'],
        ['key', '按快捷键,按键盘', 'press shortcut,hotkey', 'Press a key or key combination (e.g. ctrl+s).', 'keys:str', 'w'],
        ['scroll', '滚动,翻页', 'scroll', 'Scroll at a position.', 'x:int y:int direction:up|down amount:int', 'w'],
        ['drag', '拖动,拖拽', 'drag,drag and drop', 'Drag from one point to another.', 'x1:int y1:int x2:int y2:int', 'w'],
    ]);

    // 应用叫法只留名字（对象名不能混进来）；有些「应用」不是应用（终端、屏幕），请求里不带名字
    const ALIAS = {"unreal":["UE,UE5,虚幻,虚幻引擎","unreal,unreal engine,ue5"],"unity":["Unity,团结引擎","unity,unity editor"],"godot":["Godot","godot"],"perforce":["P4,Perforce","perforce,p4"],"gpuprof":["RenderDoc","renderdoc"],"wwise":["Wwise","wwise"],"blender":["Blender","blender"],"maya":["Maya","maya"],"houdini":["Houdini","houdini"],"substance":["Substance Painter,SP","substance painter"],"photoshop":["PS,Photoshop","photoshop"],"figma":["Figma","figma"],"ffmpeg":["FFmpeg","ffmpeg"],"imagemagick":["ImageMagick","imagemagick"],"excel":["Excel","excel"],"word":["Word","word"],"powerpoint":["PPT,PowerPoint","powerpoint"],"mail":["邮箱,Outlook,Gmail","outlook,gmail"],"calendar":["日历","calendar,google calendar,outlook calendar"],"feishu":["飞书,Lark","feishu,lark"],"notion":["Notion","notion"],"gdrive":["网盘,Google Drive","google drive"],"confluence":["Confluence","confluence"],"git":["git,本地仓库","git"],"github":["GitHub","github"],"docker":["Docker","docker"],"k8s":["K8s,Kubernetes","kubernetes,k8s"],"aws":["AWS,亚马逊云","aws"],"sql":["数据库","the database,postgres,mysql"],"redis":["Redis","redis"],"sentry":["Sentry","sentry"],"jira":["Jira","jira"],"browser":["浏览器","the browser,playwright"],"slack":["Slack","slack"],"analytics":["数据平台,运营后台","the analytics platform,the dashboard"],"support":["客服系统","the support desk"],"crowdin":["Crowdin","crowdin"],"typeform":["Typeform","typeform"],"workday":["Workday","workday"],"snowflake":["Snowflake","snowflake"],"airflow":["Airflow","airflow"],"kafka":["Kafka","kafka"],"elasticsearch":["Elasticsearch,ES","elasticsearch"],"docusign":["DocuSign","docusign"],"statuspage":["Statuspage","statuspage"],"photon":["Photon","photon"],"trello":["Trello","trello"],"mailchimp":["Mailchimp","mailchimp"]};
    for (const a of APPS) { if (ALIAS[a.ns]) { a.zh = ALIAS[a.ns][0].split(','); a.en = ALIAS[a.ns][1].split(','); } a.noApp = ["computer","shell","runtime","web","filesystem"].includes(a.ns); }

    // ═════════════════════════ 3. 展开成工具 ═════════════════════════
    function parseParams(spec) {
        const props = {}, required = [];
        for (const tok of String(spec || '').split(/\s+/).filter(Boolean)) {
            const opt = tok.endsWith('?'), t = opt ? tok.slice(0, -1) : tok, i = t.indexOf(':'), name = t.slice(0, i), ty = t.slice(i + 1);
            let p;
            if (ty === 'str') p = { type: 'string' };
            else if (ty === 'int') p = { type: 'integer' };
            else if (ty === 'num') p = { type: 'number' };
            else if (ty === 'bool') p = { type: 'boolean' };
            else if (ty === 'obj') p = { type: 'object' };
            else if (ty === 'any') p = {};
            else if (ty === 'vec3') p = { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3 };
            else if (ty === 'str[]') p = { type: 'array', items: { type: 'string' } };
            else if (ty === 'int[]') p = { type: 'array', items: { type: 'integer' } };
            else if (ty.includes('|')) p = { type: 'string', enum: ty.split('|') };
            else p = { type: 'string' };
            props[name] = p; if (!opt) required.push(name);
        }
        return { type: 'object', properties: props, required };
    }
    const ANNOT = { r: { readOnlyHint: true, openWorldHint: false }, w: { readOnlyHint: false, destructiveHint: false, openWorldHint: false }, d: { readOnlyHint: false, destructiveHint: true, openWorldHint: false }, o: { readOnlyHint: false, destructiveHint: false, openWorldHint: true } };
    const PDESC = { id: 'Name or id.', name: 'Name.', path: 'File path.', actor: 'Actor name or label.', location: 'World location [x, y, z] in cm.', rotation: 'Rotation [pitch, yaw, roll] in degrees.', scale: 'Scale [x, y, z].', intensity: 'New intensity value.', color: 'Linear RGB in 0..1.', filter: 'Substring filter.', limit: 'Max results.', cursor: 'Paging cursor from a previous call.', query: 'Search query.', code: 'Source code to execute.', format: 'Output format.', output: 'Output file path.', input: 'Input file path.', sql: 'SQL statement.', command: 'Command line.', owner: 'Repository owner.', repo: 'Repository name.' };
    const camelWords = (s) => s.replace(/_/g, ' ');
    const tools = [], byId = new Map(), apps = [];
    function addTool(app, name, zhs, ens, desc, params, kind, meta) {
        const id = `${app.ns}/${name}`; if (byId.has(id)) return null;
        const schema = parseParams(params);
        for (const [k, p] of Object.entries(schema.properties)) if (PDESC[k]) p.description = PDESC[k];
        const t = { id, ns: app.ns, name, mcpName: `mcp__${app.ns}__${name}`, title: (meta && meta.title) || name, description: `${app.name}: ${desc}`, inputSchema: schema, annotations: ANNOT[kind] || ANNOT.w, kind, zh: zhs, en: ens, obj: meta && meta.obj, verb: meta && meta.verb, phrase: !!(meta && meta.phrase), i: tools.length };
        tools.push(t); byId.set(id, t); app.tools.push(t);
        return t;
    }
    for (const app of APPS) {
        app.tools = [];
        apps.push(app);
        for (const [okey, [ozh, oen, verbs, over]] of Object.entries(app.objects)) {
            const ozhs = ozh.split(','), oens = oen.split(','), oName = oens[0], oPlural = oens[1] || oens[0] + 's';
            for (const vk of verbs.split(/\s+/).filter(Boolean)) {
                const cv = app.verbs[vk], gv = V[vk], def = cv || gv;
                if (!def) throw new Error(`unknown verb ${app.ns}.${okey}.${vk}`);
                const [vzh, ven, vdesc, vparams, kind] = def;
                // 工具名：自定义动词带前缀时（set_intensity）→ set_light_intensity；否则 verb_object
                let name;
                if (/_/.test(vk) && !/^(export|import)_/.test(vk) && !vk.startsWith(okey)) { const [head, ...rest] = vk.split('_'); name = `${head}_${okey}_${rest.join('_')}`; }
                else if (/^(export|import)_/.test(vk)) name = vk;
                else name = `${vk}_${okey}`;
                const desc = vdesc.replace(/\{o\}/g, oName).replace(/\{os\}/g, oPlural).replace(/\{app\}/g, app.name).replace(/^./, (c) => c.toUpperCase());
                const params = (over && over[vk]) || vparams;
                addTool(app, name, vzh.split(',').map((v) => ({ v, o: ozhs })), ven.split(',').map((v) => ({ v, o: [oens[0]].concat(oens.slice(2)) })), desc, params, kind, { obj: okey, verb: vk, phrase: !!cv, title: `${vk} ${camelWords(okey)}` });
            }
        }
        for (const [name, vzh, ven, desc, params, kind] of app.exact) addTool(app, name, [{ v: vzh.split(','), o: [] }], [{ v: ven.split(','), o: [] }], desc, params, kind, { exact: true, phrase: true, title: camelWords(name) });
    }
    // exact 工具：zh / en 是「整句说法」列表
    for (const t of tools) if (!t.obj) { t.zh = t.zh[0].v.map((v) => ({ v, o: [''] })); t.en = t.en[0].v.map((v) => ({ v, o: [''] })); }

    // ═════════════════════════ 4. 生成用户请求（训练 / 评测） ═════════════════════════
    // 说法模板：每条 [有应用名, 没有应用名]；{v} = 动词说法、{o} = 对象说法、{x} = 按工具参数生成的具体值（路径 / 名字 / 数字 / 文本）
    const ZH_VO = [['{app}里{v}{o}{x}', '{v}{o}{x}'], ['帮我在{app}里{v}{o}', '帮我{v}{o}'], ['把{app}里的{o}{v}', '把{o}{v}'], ['{app}：{v}{o}{x}', '{v}{o}{x}'], ['请在{app}{v}{o}{x}', '请{v}{o}{x}'], ['我要在{app}{v}{o}', '我要{v}{o}'], ['给我{v}{o}{x}', '给我{v}{o}{x}'], ['能不能帮我在{app}里{v}{o}', '能不能帮我{v}{o}'], ['用{app}{v}{o}{x}', '{v}{o}{x}']];
    const ZH_VO_HOLD = [['麻烦在{app}{v}一下{o}', '麻烦{v}一下{o}'], ['{app}那边{v}{o}{x}', '{v}{o}{x}'], ['想在{app}里{v}{o}', '想{v}{o}']];
    const ZH_PH = [['{app}里{v}{x}', '{v}{x}'], ['帮我在{app}里{v}', '帮我{v}'], ['{app}：{v}{x}', '{v}{x}'], ['能不能在{app}里{v}', '能不能{v}'], ['我想在{app}{v}', '我想{v}'], ['请{v}{x}', '请{v}{x}'], ['用{app}{v}{x}', '{v}{x}']];
    const ZH_PH_HOLD = [['麻烦在{app}{v}', '麻烦{v}'], ['{app}那边{v}{x}', '{v}{x}']];
    const EN_VO = [['{v} the {o} in {app}{x}', '{v} the {o}{x}'], ['{app}: {v} a {o}{x}', '{v} a {o}{x}'], ['please {v} the {o}{x}', 'please {v} the {o}{x}'], ['can you {v} a {o} in {app}', 'can you {v} a {o}'], ['I need to {v} the {o}', 'I need to {v} the {o}']];
    const EN_VO_HOLD = [['could you {v} the {o} for me in {app}', 'could you {v} the {o} for me'], ['go ahead and {v} the {o}{x}', 'go ahead and {v} the {o}{x}']];
    const EN_PH = [['{v} in {app}{x}', '{v}{x}'], ['please {v}{x}', 'please {v}{x}'], ['can you {v} in {app}', 'can you {v}'], ['{app}: {v}{x}', '{v}{x}'], ['I need to {v}', 'I need to {v}']];
    const EN_PH_HOLD = [['could you {v} for me in {app}', 'could you {v} for me'], ['go ahead and {v}{x}', 'go ahead and {v}{x}']];
    // 只读工具的提问式说法（查数据、看状态、列清单的请求常常是问句）
    const ZH_Q = [['{app}里的{o}现在是多少{x}', '{o}现在是多少{x}'], ['查一下{app}的{o}{x}', '查一下{o}{x}'], ['{app}里有多少{o}', '有多少{o}'], ['{o}是什么情况{x}', '{o}是什么情况{x}'], ['看看{app}里的{o}{x}', '看看{o}{x}']];
    const ZH_QP = [['{app}里{v}是多少{x}', '{v}是多少{x}'], ['查一下{app}的{v}{x}', '查一下{v}{x}'], ['{v}怎么样了{x}', '{v}怎么样了{x}']];
    const EN_Q = [['how many {o} are there in {app}', 'how many {o} are there'], ['what is the current {o} in {app}{x}', 'what is the current {o}{x}'], ['show me the {o}{x}', 'show me the {o}{x}']];
    const EN_QP = [['what is the {v} in {app}{x}', 'what is the {v}{x}'], ['check the {v}{x}', 'check the {v}{x}']];
    const NO_BA = new Set(['list', 'get', 'search', 'query', 'find', 'create', 'add', 'logs', 'read', 'schedule', 'invite', 'play', 'sync', 'set', 'insert', 'reference', 'build', 'run', 'start']);   // 「把 X 列出」这类说法不通
    /** {x}：按工具参数给一个具体值（训练时让模型见过「带参数的请求」；第 19 篇的参数抽取也从这里学到格式） */
    function argSnippet(t, zh, rand) {
        const P = t.inputSchema.properties, R = new Set(t.inputSchema.required), has = (k) => R.has(k), num = Object.keys(P).find((k) => R.has(k) && (P[k].type === 'number' || P[k].type === 'integer')), pick = (a) => a[Math.floor(rand() * a.length)];   // 只用必填参数（limit / cursor 这类分页参数不出现在请求里）
        const opts = [];
        if (has('path') || has('output') || has('destination')) opts.push(zh ? pick([' 到 D:/Export/hero.fbx', ' 存到 out/level_01.png', ' 路径是 /Game/Props']) : pick([' to D:/Export/hero.fbx', ' at out/level_01.png', ' into /Game/Props']));
        if (has('name') || has('new_name') || has('title')) opts.push(zh ? pick(['，叫 Hero', '，名字是 test_01', '，标题写“版本更新”']) : pick([' named Hero', ' called test_01', ' titled "release notes"']));
        if (num) opts.push(zh ? pick([' 到 5000', ' 改成 0.5', '，数量 3', ' 设为 120']) : pick([' to 5000', ' to 0.5', ' with count 3', ' set to 120']));
        if (has('start_date') || has('cohort_start') || has('date')) opts.push(zh ? pick(['，昨天的', '，上周的', '，最近 7 天', '，2024-06-01 到 2024-06-07', '，这个月']) : pick([' for yesterday', ' for last week', ' over the last 7 days', ' from 2024-06-01 to 2024-06-07']));
        if (has('content') || has('body') || has('text') || has('query')) opts.push(zh ? pick(['：“今晚 8 点停服维护”', '，内容是“构建好了”']) : pick([': "maintenance at 8 pm"', ' saying "build is ready"']));
        return opts.length && rand() < 0.45 ? pick(opts) : '';
    }
    function fill(tpl, app, v, o, x) {
        const out = (app ? tpl[0] : tpl[1]).replace('{app}', app).replace('{v}', v).replace('{o}', o).replace('{x}', x);
        return out.replace(/\s+/g, ' ').replace(/的的/g, '的').trim();
    }
    const appOf = new Map();
    /** 给一个工具生成 n 条请求；held = true 用留出的模板（评测「没见过的说法」）*/
    function utterances(t, rand, n, held) {
        let app = appOf.get(t.ns); if (!app) { app = apps.find((q) => q.ns === t.ns); appOf.set(t.ns, app); }
        const out = [], pick = (arr) => arr[Math.floor(rand() * arr.length)];
        for (let k = 0; k < n; k++) {
            const zh = rand() < 0.62, f = pick(zh ? t.zh : t.en), v = Array.isArray(f.v) ? pick(f.v) : f.v, o = f.o && f.o.length ? pick(f.o) : '';
            const appName = !app.noApp && rand() < 0.65 ? pick(zh ? app.zh : app.en) : '';
            const x = argSnippet(t, zh, rand);
            let text;
            if (t.annotations.readOnlyHint && rand() < 0.3) {   // 只读工具：三成用提问式
                text = o && !t.phrase ? fill(pick(zh ? ZH_Q : EN_Q), appName, v, o, x) : fill(pick(zh ? ZH_QP : EN_QP), appName, v, '', x);
            } else if (!o || (t.phrase && rand() < 0.45)) {   // 短语动词本身就是一句话（偶尔加上对象当话题）；自定义动词一半时间也按「动词 + 对象」说
                const core = o && rand() < 0.3 ? (zh ? o + v : `${v} for the ${o}`) : v;
                text = fill(pick(zh ? (held ? ZH_PH_HOLD : ZH_PH) : (held ? EN_PH_HOLD : EN_PH)), appName, core, '', x);
            } else {
                let tpls = zh ? (held ? ZH_VO_HOLD : ZH_VO) : (held ? EN_VO_HOLD : EN_VO);
                if (zh && NO_BA.has(t.verb)) tpls = tpls.filter((q) => !q[0].includes('把'));
                text = fill(pick(tpls), appName, v, o, x);
            }
            out.push({ text, tool: t.id, ns: t.ns, zh, held: !!held });
        }
        return out;
    }
    // 「新工具」：每个应用留出约 8% 的工具不进训练（只有描述），检验检索能否找到从没见过请求的工具
    function isNewTool(t) { let h = 2166136261; for (const c of t.id) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); } return ((h >>> 0) % 100) < 8; }

    // ═════════════════════════ 5. 动作类型（第 19 篇最后一步）：直接回答 / 调一个工具 / 写段代码 / 操作界面 / 先问清楚 ═════════════════════════
    const ACTIONS = ['answer', 'tool', 'code', 'gui', 'ask'];
    const ACT_ZH = { answer: '直接回答', tool: '调一个工具', code: '写一段代码（批量 / 循环 / 统计）', gui: '操作界面（Computer Use）', ask: '先问清楚' };
    const CONCEPTS = ['蓝图', 'PBR 材质', 'LOD', 'Nanite', 'Lumen', '光照贴图', '法线贴图', 'UV', 'Draw Call', '预制体', '协程', 'ECS', 'git rebase', 'Docker 镜像', 'Kubernetes', 'SQL 索引', '留存率', 'LTV', 'A/B 测试', 'FBX 和 glTF 的区别', 'HDR', 'Gamma 校正', '四元数', '蒙皮'];
    const CONCEPTS_EN = ['blueprints', 'PBR materials', 'LOD', 'Nanite', 'lightmaps', 'normal maps', 'draw calls', 'prefabs', 'git rebase', 'docker images', 'retention', 'quaternions', 'skinning'];
    function actionSamples(rand, n) {
        const pick = (a) => a[Math.floor(rand() * a.length)], out = [];
        for (let i = 0; i < n; i++) {
            const kind = ACTIONS[i % ACTIONS.length], zh = rand() < 0.65;
            let text;
            if (kind === 'answer') text = zh ? pick(['什么是{c}', '{c}是什么意思', '解释一下{c}', '为什么要用{c}', '{c}有什么用', '讲讲{c}的原理']).replace('{c}', pick(CONCEPTS)) : pick(['what is {c}', 'explain {c}', 'why do we need {c}', 'how does {c} work']).replace('{c}', pick(CONCEPTS_EN));
            else {
                const t = tools[Math.floor(rand() * tools.length)], u = utterances(t, rand, 1, false)[0].text;
                if (kind === 'tool') text = u;
                else if (kind === 'code') {
                    const f = zh ? t.zh[0] : t.en[0], v = Array.isArray(f.v) ? f.v[0] : f.v, o = f.o && f.o[0] ? f.o[0] : (zh ? '东西' : 'items');
                    text = zh ? pick(['把所有{o}都{v}', '批量{v}{o}', '每个{o}都{v}一遍', '统计每个关卡有多少{o}', '把名字带 old 的{o}全部{v}', '遍历所有{o}，满足条件的就{v}', '{v}全部 300 个{o}']).replace('{o}', o).replace('{v}', v) : pick(['{v} every {o}', 'for each {o}, {v} it', 'batch {v} all {o}', 'count how many {o} are in each level', '{v} all 300 {o}']).replace('{o}', o).replace('{v}', v);
                } else if (kind === 'gui') text = zh ? pick(['在这个软件的界面上点一下{b}按钮', '这个后台没有接口，帮我在网页上点{b}', '打开菜单找到{b}然后点击', '在窗口右上角点{b}', '这个老工具只能手动点，帮我把{b}点了']).replace('{b}', pick(['导出', '提交', '确定', '发布', '保存设置', '刷新'])) : pick(['click the {b} button in this app window', 'there is no API, just click {b} on the screen', 'open the menu and press {b}']).replace('{b}', pick(['Export', 'Submit', 'OK', 'Publish']));
                else text = zh ? pick(['帮我改一下那个', '弄一下', '处理一下这个', '把它修好', '上次那个再来一遍', '优化一下', '搞定它', '看看怎么回事']) : pick(['fix it', 'do that thing again', 'handle this', 'make it better', 'can you sort it out']);
            }
            out.push({ text, action: kind, zh });
        }
        return out;
    }

    return { V, APPS: apps, tools, byId, parseParams, utterances, isNewTool, ACTIONS, ACT_ZH, actionSamples };
})();
if (typeof module !== 'undefined' && module.exports) module.exports = TKCAT;
