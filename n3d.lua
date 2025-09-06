local SCREEN_WIDTH, SCREEN_HEIGHT = screen.getSize()
local SCREEN_CENTER_X = SCREEN_WIDTH / 2
local SCREEN_CENTER_Y = SCREEN_HEIGHT / 2

local function degToRad(deg) return deg * math.pi / 180 end
local function clamp(value, min, max) return math.max(min, math.min(max, value)) end
local function lerp(a, b, t) return a + (b - a) * t end

Vector3 = {}
Vector3.__index = Vector3

function Vector3.new(x, y, z)
    return setmetatable({x = x or 0, y = y or 0, z = z or 0}, Vector3)
end

function Vector3:add(other) return Vector3.new(self.x + other.x, self.y + other.y, self.z + other.z) end
function Vector3:subtract(other) return Vector3.new(self.x - other.x, self.y - other.y, self.z - other.z) end
function Vector3:multiply(scalar) return Vector3.new(self.x * scalar, self.y * scalar, self.z * scalar) end
function Vector3:divide(scalar) return scalar ~= 0 and Vector3.new(self.x / scalar, self.y / scalar, self.z / scalar) or Vector3.new(0, 0, 0) end
function Vector3:dot(other) return self.x * other.x + self.y * other.y + self.z * other.z end
function Vector3:cross(other) return Vector3.new(self.y * other.z - self.z * other.y, self.z * other.x - self.x * other.z, self.x * other.y - self.y * other.x) end
function Vector3:length() return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z) end
function Vector3:lengthSquared() return self.x * self.x + self.y * self.y + self.z * self.z end
function Vector3:normalize() local len = self:length(); return len > 0 and self:divide(len) or Vector3.new(0, 0, 0) end
function Vector3:reflect(normal) return self:subtract(normal:multiply(2 * self:dot(normal))) end
function Vector3:lerp(other, t) return Vector3.new(lerp(self.x, other.x, t), lerp(self.y, other.y, t), lerp(self.z, other.z, t)) end

Matrix4 = {}
Matrix4.__index = Matrix4

function Matrix4.new()
    local m = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
    return setmetatable({m = m}, Matrix4)
end

function Matrix4.identity()
    local matrix = Matrix4.new()
    matrix.m[1], matrix.m[6], matrix.m[11], matrix.m[16] = 1, 1, 1, 1
    return matrix
end

function Matrix4.translation(x, y, z)
    local matrix = Matrix4.identity()
    matrix.m[4], matrix.m[8], matrix.m[12] = x, y, z
    return matrix
end

function Matrix4.scaling(x, y, z)
    local matrix = Matrix4.identity()
    matrix.m[1], matrix.m[6], matrix.m[11] = x, y, z
    return matrix
end

function Matrix4.rotationX(angle)
    local matrix = Matrix4.identity()
    local c, s = math.cos(angle), math.sin(angle)
    matrix.m[6], matrix.m[7], matrix.m[10], matrix.m[11] = c, -s, s, c
    return matrix
end

function Matrix4.rotationY(angle)
    local matrix = Matrix4.identity()
    local c, s = math.cos(angle), math.sin(angle)
    matrix.m[1], matrix.m[3], matrix.m[9], matrix.m[11] = c, s, -s, c
    return matrix
end

function Matrix4.rotationZ(angle)
    local matrix = Matrix4.identity()
    local c, s = math.cos(angle), math.sin(angle)
    matrix.m[1], matrix.m[2], matrix.m[5], matrix.m[6] = c, -s, s, c
    return matrix
end

function Matrix4:multiply(other)
    local result = Matrix4.new()
    for row = 0, 3 do
        for col = 0, 3 do
            local sum = 0
            for i = 0, 3 do
                sum = sum + self.m[row * 4 + i + 1] * other.m[i * 4 + col + 1]
            end
            result.m[row * 4 + col + 1] = sum
        end
    end
    return result
end

function Matrix4:transformNormal(normal)
    local x = normal.x * self.m[1] + normal.y * self.m[2] + normal.z * self.m[3]
    local y = normal.x * self.m[5] + normal.y * self.m[6] + normal.z * self.m[7]
    local z = normal.x * self.m[9] + normal.y * self.m[10] + normal.z * self.m[11]
    return Vector3.new(x, y, z):normalize()
end

function Matrix4:transformVector(v)
    local x = v.x * self.m[1] + v.y * self.m[2] + v.z * self.m[3] + self.m[4]
    local y = v.x * self.m[5] + v.y * self.m[6] + v.z * self.m[7] + self.m[8]
    local z = v.x * self.m[9] + v.y * self.m[10] + v.z * self.m[11] + self.m[12]
    local w = v.x * self.m[13] + v.y * self.m[14] + v.z * self.m[15] + self.m[16]
    return w ~= 0 and Vector3.new(x/w, y/w, z/w) or Vector3.new(x, y, z)
end

function Matrix4.lookAt(eye, target, up)
    local zAxis = eye:subtract(target):normalize()
    local xAxis = up:cross(zAxis):normalize()
    local yAxis = zAxis:cross(xAxis)

    local matrix = Matrix4.new()
    matrix.m[1], matrix.m[2], matrix.m[3] = xAxis.x, yAxis.x, zAxis.x
    matrix.m[5], matrix.m[6], matrix.m[7] = xAxis.y, yAxis.y, zAxis.y
    matrix.m[9], matrix.m[10], matrix.m[11] = xAxis.z, yAxis.z, zAxis.z
    matrix.m[4], matrix.m[8], matrix.m[12] = -xAxis:dot(eye), -yAxis:dot(eye), -zAxis:dot(eye)
    matrix.m[16] = 1

    return matrix
end

Material = {}
Material.__index = Material

function Material.new(diffuse, specular, shininess, ambient, emission)
    return setmetatable({
        diffuse = diffuse or {r = 200, g = 200, b = 200},
        specular = specular or {r = 255, g = 255, b = 255},
        shininess = shininess or 32.0,
        ambient = ambient or {r = 50, g = 50, b = 50},
        emission = emission or {r = 0, g = 0, b = 0}
    }, Material)
end

Light = {}
Light.__index = Light

function Light.new(type, properties)
    local light = setmetatable({type = type}, Light)
    for k, v in pairs(properties) do light[k] = v end
    return light
end

function Light.directional(direction, color, intensity)
    return Light.new("directional", {
        direction = direction:normalize(),
        color = color or {r = 255, g = 255, b = 255},
        intensity = intensity or 1.0
    })
end

function Light.point(position, color, intensity, range, attenuation)
    return Light.new("point", {
        position = position,
        color = color or {r = 255, g = 255, b = 255},
        intensity = intensity or 1.0,
        range = range or 10.0,
        attenuation = attenuation or {constant = 1.0, linear = 0.09, quadratic = 0.032}
    })
end

function Light.spot(position, direction, color, intensity, cutoff, outerCutoff, range, attenuation)
    return Light.new("spot", {
        position = position,
        direction = direction:normalize(),
        color = color or {r = 255, g = 255, b = 255},
        intensity = intensity or 1.0,
        cutoff = math.cos(degToRad(cutoff or 12.5)),
        outerCutoff = math.cos(degToRad(outerCutoff or 17.5)),
        range = range or 15.0,
        attenuation = attenuation or {constant = 1.0, linear = 0.09, quadratic = 0.032}
    })
end

function Light.ambient(color, intensity)
    return Light.new("ambient", {
        color = color or {r = 50, g = 50, b = 50},
        intensity = intensity or 0.1
    })
end

Mesh = {}
Mesh.__index = Mesh

function Mesh.new()
    return setmetatable({vertices = {}, triangles = {}}, Mesh)
end

function Mesh:addVertex(position, normal, texcoord)
    table.insert(self.vertices, {
        position = position or Vector3.new(0, 0, 0), 
        normal = normal or Vector3.new(0, 1, 0),
        texcoord = texcoord or {u = 0, v = 0}
    })
    return #self.vertices
end

function Mesh:addTriangle(i1, i2, i3, material)
    table.insert(self.triangles, {
        v1 = self.vertices[i1], 
        v2 = self.vertices[i2], 
        v3 = self.vertices[i3],
        material = material,
        indices = {i1, i2, i3}
    })
end

function Mesh:addTriangleRaw(v1, v2, v3, material)

    local i1 = self:addVertex(v1.position, v1.normal, v1.texcoord)
    local i2 = self:addVertex(v2.position, v2.normal, v2.texcoord)
    local i3 = self:addVertex(v3.position, v3.normal, v3.texcoord)
    self:addTriangle(i1, i2, i3, material)
end

function Mesh:calculateNormals()

    for _, vertex in ipairs(self.vertices) do
        vertex.normal = Vector3.new(0, 0, 0)
    end

    for _, triangle in ipairs(self.triangles) do
        local edge1 = triangle.v2.position:subtract(triangle.v1.position)
        local edge2 = triangle.v3.position:subtract(triangle.v1.position)
        local faceNormal = edge1:cross(edge2):normalize()

        triangle.v1.normal = triangle.v1.normal:add(faceNormal)
        triangle.v2.normal = triangle.v2.normal:add(faceNormal)
        triangle.v3.normal = triangle.v3.normal:add(faceNormal)
    end

    for _, vertex in ipairs(self.vertices) do
        vertex.normal = vertex.normal:normalize()
    end
end

function Mesh.fromOBJ(objString, material, scale)
    local mesh = Mesh.new()
    scale = scale or 1.0

    local vertices = {}
    local normals = {}
    local texcoords = {}

    for line in objString:gmatch("[^\r\n]+") do
        line = line:gsub("^%s+", ""):gsub("%s+$", "") 

        if line:sub(1, 2) == "v " then

            local x, y, z = line:match("v%s+([%-%d%.]+)%s+([%-%d%.]+)%s+([%-%d%.]+)")
            if x and y and z then
                table.insert(vertices, Vector3.new(tonumber(x) * scale, tonumber(y) * scale, tonumber(z) * scale))
            end

        elseif line:sub(1, 3) == "vn " then

            local x, y, z = line:match("vn%s+([%-%d%.]+)%s+([%-%d%.]+)%s+([%-%d%.]+)")
            if x and y and z then
                table.insert(normals, Vector3.new(tonumber(x), tonumber(y), tonumber(z)))
            end

        elseif line:sub(1, 3) == "vt " then

            local u, v = line:match("vt%s+([%-%d%.]+)%s+([%-%d%.]+)")
            if u and v then
                table.insert(texcoords, {u = tonumber(u), v = tonumber(v)})
            end

        elseif line:sub(1, 2) == "f " then

            local faceVertices = {}

            for vertexData in line:gmatch("%S+") do
                if vertexData ~= "f" then
                    local parts = {}
                    for part in vertexData:gmatch("[^%/]+") do
                        table.insert(parts, part)
                    end

                    local vi, ti, ni

                    if #parts == 1 then
                        vi = parts[1]
                    elseif #parts == 2 then
                        vi, ni = parts[1], parts[2]
                    else
                        vi, ti, ni = parts[1], parts[2], parts[3]
                    end

                    if vi then
                        local vertex = {
                            position = vertices[tonumber(vi)],
                            normal = ni and normals[tonumber(ni)] or Vector3.new(0, 1, 0),
                            texcoord = ti and texcoords[tonumber(ti)] or {u = 0, v = 0}
                        }
                        table.insert(faceVertices, vertex)
                    end
                end
            end

            if #faceVertices >= 3 then
                for i = 2, #faceVertices - 1 do
                    mesh:addTriangleRaw(faceVertices[1], faceVertices[i], faceVertices[i + 1], material)
                end
            end
        end
    end

    if #normals == 0 then
        mesh:calculateNormals()
    end

    return mesh
end

function Mesh.cube(size, material)
    size = size or 1
    local mesh = Mesh.new()

    local v1 = mesh:addVertex(Vector3.new(-size, -size, -size), Vector3.new(0, 0, -1))
    local v2 = mesh:addVertex(Vector3.new(size, -size, -size), Vector3.new(0, 0, -1))
    local v3 = mesh:addVertex(Vector3.new(size, size, -size), Vector3.new(0, 0, -1))
    local v4 = mesh:addVertex(Vector3.new(-size, size, -size), Vector3.new(0, 0, -1))

    local v5 = mesh:addVertex(Vector3.new(-size, -size, size), Vector3.new(0, 0, 1))
    local v6 = mesh:addVertex(Vector3.new(size, -size, size), Vector3.new(0, 0, 1))
    local v7 = mesh:addVertex(Vector3.new(size, size, size), Vector3.new(0, 0, 1))
    local v8 = mesh:addVertex(Vector3.new(-size, size, size), Vector3.new(0, 0, 1))

    local v9 = mesh:addVertex(Vector3.new(-size, -size, size), Vector3.new(-1, 0, 0))
    local v10 = mesh:addVertex(Vector3.new(-size, -size, -size), Vector3.new(-1, 0, 0))
    local v11 = mesh:addVertex(Vector3.new(-size, size, -size), Vector3.new(-1, 0, 0))
    local v12 = mesh:addVertex(Vector3.new(-size, size, size), Vector3.new(-1, 0, 0))

    local v13 = mesh:addVertex(Vector3.new(size, -size, -size), Vector3.new(1, 0, 0))
    local v14 = mesh:addVertex(Vector3.new(size, -size, size), Vector3.new(1, 0, 0))
    local v15 = mesh:addVertex(Vector3.new(size, size, size), Vector3.new(1, 0, 0))
    local v16 = mesh:addVertex(Vector3.new(size, size, -size), Vector3.new(1, 0, 0))

    local v17 = mesh:addVertex(Vector3.new(-size, size, -size), Vector3.new(0, 1, 0))
    local v18 = mesh:addVertex(Vector3.new(size, size, -size), Vector3.new(0, 1, 0))
    local v19 = mesh:addVertex(Vector3.new(size, size, size), Vector3.new(0, 1, 0))
    local v20 = mesh:addVertex(Vector3.new(-size, size, size), Vector3.new(0, 1, 0))

    local v21 = mesh:addVertex(Vector3.new(-size, -size, size), Vector3.new(0, -1, 0))
    local v22 = mesh:addVertex(Vector3.new(size, -size, size), Vector3.new(0, -1, 0))
    local v23 = mesh:addVertex(Vector3.new(size, -size, -size), Vector3.new(0, -1, 0))
    local v24 = mesh:addVertex(Vector3.new(-size, -size, -size), Vector3.new(0, -1, 0))

    mesh:addTriangle(v1, v2, v3, material)
    mesh:addTriangle(v1, v3, v4, material)

    mesh:addTriangle(v6, v5, v8, material)
    mesh:addTriangle(v6, v8, v7, material)

    mesh:addTriangle(v9, v10, v11, material)
    mesh:addTriangle(v9, v11, v12, material)

    mesh:addTriangle(v13, v14, v15, material)
    mesh:addTriangle(v13, v15, v16, material)

    mesh:addTriangle(v17, v18, v19, material)
    mesh:addTriangle(v17, v19, v20, material)

    mesh:addTriangle(v21, v22, v23, material)
    mesh:addTriangle(v21, v23, v24, material)

    return mesh
end

function Mesh.sphere(radius, segments, material)
    radius = radius or 1
    segments = segments or 16
    local mesh = Mesh.new()

    local vertices = {}
    for i = 0, segments do
        local lat = math.pi * i / segments
        local sinLat, cosLat = math.sin(lat), math.cos(lat)

        for j = 0, segments do
            local lon = 2 * math.pi * j / segments
            local sinLon, cosLon = math.sin(lon), math.cos(lon)

            local x = cosLon * sinLat * radius
            local y = cosLat * radius
            local z = sinLon * sinLat * radius

            local normal = Vector3.new(x, y, z):normalize()
            table.insert(vertices, mesh:addVertex(Vector3.new(x, y, z), normal))
        end
    end

    for i = 0, segments - 1 do
        for j = 0, segments - 1 do
            local first = i * (segments + 1) + j + 1
            local second = first + segments + 1

            mesh:addTriangle(first, second, first + 1, material)
            mesh:addTriangle(second, second + 1, first + 1, material)
        end
    end

    return mesh
end

function Mesh.plane(size, y, material)
    size = size or 10
    y = y or 0
    local mesh = Mesh.new()
    local half = size / 2

    local v1 = mesh:addVertex(Vector3.new(-half, y, -half), Vector3.new(0, 1, 0))
    local v2 = mesh:addVertex(Vector3.new(half, y, -half), Vector3.new(0, 1, 0))
    local v3 = mesh:addVertex(Vector3.new(half, y, half), Vector3.new(0, 1, 0))
    local v4 = mesh:addVertex(Vector3.new(-half, y, half), Vector3.new(0, 1, 0))

    mesh:addTriangle(v1, v2, v3, material)
    mesh:addTriangle(v1, v3, v4, material)

    return mesh
end

SceneObject = {}
SceneObject.__index = SceneObject

function SceneObject.new(mesh, material, position, rotation, scale, castsShadow, receivesShadow)
    return setmetatable({
        mesh = mesh,
        material = material,
        position = position or Vector3.new(0, 0, 0),
        rotation = rotation or Vector3.new(0, 0, 0),
        scale = scale or Vector3.new(1, 1, 1),
        transform = Matrix4.identity(),
        needsUpdate = true,
        castsShadow = castsShadow ~= false,
        receivesShadow = receivesShadow ~= false
    }, SceneObject)
end

function SceneObject:updateTransform()
    if not self.needsUpdate then return self.transform end

    local scaleMat = Matrix4.scaling(self.scale.x, self.scale.y, self.scale.z)
    local rotXMat = Matrix4.rotationX(self.rotation.x)
    local rotYMat = Matrix4.rotationY(self.rotation.y)
    local rotZMat = Matrix4.rotationZ(self.rotation.z)
    local transMat = Matrix4.translation(self.position.x, self.position.y, self.position.z)

    local rotation = rotXMat:multiply(rotYMat):multiply(rotZMat)
    self.transform = transMat:multiply(rotation):multiply(scaleMat)
    self.needsUpdate = false

    return self.transform
end

function SceneObject:setPosition(x, y, z)
    self.position = Vector3.new(x, y, z)
    self.needsUpdate = true
end

function SceneObject:setRotation(x, y, z)
    self.rotation = Vector3.new(x, y, z)
    self.needsUpdate = true
end

function SceneObject:setScale(x, y, z)
    self.scale = Vector3.new(x, y, z)
    self.needsUpdate = true
end

Camera = {}
Camera.__index = Camera

function Camera.new(position, target, up, fov, near, far)
    return setmetatable({
        position = position or Vector3.new(0, 0, 10),
        target = target or Vector3.new(0, 0, 0),
        up = up or Vector3.new(0, 1, 0),
        fov = fov or 70,
        near = near or 0.1,
        far = far or 100.0,
        viewMatrix = Matrix4.identity(),
        projMatrix = Matrix4.identity(),
        needsUpdate = true
    }, Camera)
end

function Camera:updateMatrices(aspectRatio)
    if self.needsUpdate then
        self.viewMatrix = Matrix4.lookAt(self.position, self.target, self.up)

        local fovRad = degToRad(self.fov)
        local f = 1.0 / math.tan(fovRad / 2)

        self.projMatrix = Matrix4.identity()
        self.projMatrix.m[1] = f / aspectRatio
        self.projMatrix.m[6] = f
        self.projMatrix.m[11] = (self.far + self.near) / (self.near - self.far)
        self.projMatrix.m[12] = (2 * self.far * self.near) / (self.near - self.far)
        self.projMatrix.m[15] = -1
        self.projMatrix.m[16] = 0

        self.needsUpdate = false
    end

    return self.viewMatrix, self.projMatrix
end

function Camera:setPosition(x, y, z)
    self.position = Vector3.new(x, y, z)
    self.needsUpdate = true
end

function Camera:setTarget(x, y, z)
    self.target = Vector3.new(x, y, z)
    self.needsUpdate = true
end

Scene = {}
Scene.__index = Scene

function Scene.new()
    return setmetatable({
        objects = {},
        lights = {},
        camera = Camera.new(),
        ambientLight = {r = 30, g = 30, b = 30}
    }, Scene)
end

function Scene:addObject(object)
    table.insert(self.objects, object)
end

function Scene:addLight(light)
    table.insert(self.lights, light)
end

function Scene:setCamera(camera)
    self.camera = camera
end

TriangleRenderer = {}
TriangleRenderer.__index = TriangleRenderer

function TriangleRenderer.new()
    return setmetatable({
        zBuffer = {},
        initialized = false
    }, TriangleRenderer)
end

function TriangleRenderer:initZBuffer()

    if not self.zBuffer then
        self.zBuffer = {}
    end

    for y = 1, SCREEN_HEIGHT do
        if not self.zBuffer[y] then
            self.zBuffer[y] = {}
        end
        for x = 1, SCREEN_WIDTH do
            if not self.zBuffer[y][x] then
                self.zBuffer[y][x] = 100.0
            end
        end
    end
    self.initialized = true
end

function TriangleRenderer:clearZBuffer()

    self:initZBuffer()
    for y = 1, SCREEN_HEIGHT do
        for x = 1, SCREEN_WIDTH do
            self.zBuffer[y][x] = 100.0
        end
    end
end

function TriangleRenderer:drawTriangle(v1, v2, v3, material, lights, camera)
    self:initZBuffer()

    local function simpleProject(vertex)
        local viewMatrix, projMatrix = camera:updateMatrices(SCREEN_WIDTH / SCREEN_HEIGHT)

        local viewPos = viewMatrix:transformVector(vertex.position)
        if viewPos.z >= -0.1 then return nil end

        local projPos = projMatrix:transformVector(viewPos)
        if projPos.z == 0 then return nil end

        local ndcX, ndcY = projPos.x / projPos.z, projPos.y / projPos.z
        local screenX = math.floor(SCREEN_CENTER_X + ndcX * SCREEN_CENTER_X)
        local screenY = math.floor(SCREEN_CENTER_Y - ndcY * SCREEN_CENTER_Y)

        return {
            x = clamp(screenX, 0, SCREEN_WIDTH - 1),
            y = clamp(screenY, 0, SCREEN_HEIGHT - 1),
            z = -viewPos.z
        }
    end

    local p1 = simpleProject(v1)
    local p2 = simpleProject(v2)
    local p3 = simpleProject(v3)

    if not (p1 and p2 and p3) then return end

    local edge1 = v2.position:subtract(v1.position)
    local edge2 = v3.position:subtract(v1.position)
    local normal = edge1:cross(edge2):normalize()

    local center = Vector3.new(
        (v1.position.x + v2.position.x + v3.position.x) / 3,
        (v1.position.y + v2.position.y + v3.position.z) / 3,
        (v1.position.z + v2.position.z + v3.position.z) / 3
    )

    local viewDir = camera.position:subtract(center):normalize()

    local color = {r = 128, g = 128, b = 128}
    if material then
        color = material.diffuse

        if lights and #lights > 0 then
            for _, light in ipairs(lights) do
                if light.type == "directional" then
                    local lightDir = light.direction:multiply(-1):normalize()
                    local diff = math.max(normal:dot(lightDir), 0.0)

                    color.r = clamp(math.floor(color.r * diff * light.intensity), 0, 255)
                    color.g = clamp(math.floor(color.g * diff * light.intensity), 0, 255)
                    color.b = clamp(math.floor(color.b * diff * light.intensity), 0, 255)
                    break
                end
            end
        end
    end

    local points = {p1, p2, p3}
    table.sort(points, function(a, b) return a.y < b.y end)
    local top, mid, bot = points[1], points[2], points[3]

    screen.setColor(color.r, color.g, color.b)

    local function interpY(y, a, b)
        if b.y == a.y then return {x = a.x, z = a.z} end
        local t = (y - a.y) / (b.y - a.y)
        return {x = lerp(a.x, b.x, t), z = lerp(a.z, b.z, t)}
    end

    local function drawSpan(y, left, right)
        if left.x > right.x then left, right = right, left end
        local startX = math.max(0, math.floor(left.x))
        local endX = math.min(SCREEN_WIDTH - 1, math.floor(right.x))

        for x = startX, endX do
            local t = (right.x > left.x) and ((x - left.x) / (right.x - left.x)) or 0
            local depth = lerp(left.z, right.z, t)

            if depth < self.zBuffer[y + 1][x + 1] then
                self.zBuffer[y + 1][x + 1] = depth
                screen.drawPixel(x, y)
            end
        end
    end

    for y = math.max(0, top.y), math.min(SCREEN_HEIGHT - 1, mid.y) do
        local left = interpY(y, top, bot)
        local right = interpY(y, top, mid)
        drawSpan(y, left, right)
    end

    for y = math.max(0, mid.y + 1), math.min(SCREEN_HEIGHT - 1, bot.y) do
        local left = interpY(y, top, bot)
        local right = interpY(y, mid, bot)
        drawSpan(y, left, right)
    end
end

ShadowSystem = {}
ShadowSystem.__index = ShadowSystem

function ShadowSystem.new()
    return setmetatable({
        shadowPlanes = {},
        shadowIntensity = 0.7,
        softShadowSamples = 4,
        biasEpsilon = 0.005,
        maxShadowDistance = 50
    }, ShadowSystem)
end

function ShadowSystem:addShadowPlane(plane)
    table.insert(self.shadowPlanes, plane)
end

function ShadowSystem:calculateShadowProjection(point, lightSource, planeY, planeNormal)
    planeNormal = planeNormal or Vector3.new(0, 1, 0)

    if lightSource.type == "point" then

        local lightDir = point:subtract(lightSource.position)
        local distance = lightDir:length()

        if distance > lightSource.range then return nil end

        lightDir = lightDir:normalize()

        local denominator = lightDir:dot(planeNormal)
        if math.abs(denominator) < 1e-6 then return nil end

        local t = (planeY - lightSource.position.y) / lightDir.y
        if t <= 0 then return nil end 

        local projectedPoint = lightSource.position:add(lightDir:multiply(t))
        projectedPoint.y = planeY + self.biasEpsilon

        local attenuationFactor = 1.0 / (1.0 + 0.05 * distance + 0.01 * distance * distance)
        local intensity = self.shadowIntensity * attenuationFactor

        return {
            position = projectedPoint,
            intensity = intensity,
            size = math.min(2.0, 0.5 + distance * 0.1) 
        }

    elseif lightSource.type == "directional" then

        local lightDir = lightSource.direction:normalize()

        local denominator = lightDir:dot(planeNormal)
        if math.abs(denominator) < 1e-6 then return nil end

        local t = (planeY - point.y) / lightDir.y
        if t <= 0 then return nil end

        local projectedPoint = point:add(lightDir:multiply(t))
        projectedPoint.y = planeY + self.biasEpsilon

        return {
            position = projectedPoint,
            intensity = self.shadowIntensity,
            size = 1.0 
        }

    elseif lightSource.type == "spot" then

        local lightDir = point:subtract(lightSource.position)
        local distance = lightDir:length()

        if distance > lightSource.range then return nil end

        lightDir = lightDir:normalize()

        local spotDir = lightSource.direction:normalize()
        local theta = lightDir:multiply(-1):dot(spotDir)

        if theta < lightSource.outerCutoff then return nil end

        local denominator = lightDir:dot(planeNormal)
        if math.abs(denominator) < 1e-6 then return nil end

        local t = (planeY - lightSource.position.y) / lightDir.y
        if t <= 0 then return nil end

        local projectedPoint = lightSource.position:add(lightDir:multiply(t))
        projectedPoint.y = planeY + self.biasEpsilon

        local epsilon = lightSource.cutoff - lightSource.outerCutoff
        local spotIntensity = clamp((theta - lightSource.outerCutoff) / epsilon, 0.0, 1.0)
        local attenuationFactor = 1.0 / (1.0 + 0.05 * distance + 0.01 * distance * distance)

        return {
            position = projectedPoint,
            intensity = self.shadowIntensity * spotIntensity * attenuationFactor,
            size = math.min(2.5, 0.3 + distance * 0.15)
        }
    end

    return nil
end

function ShadowSystem:generateShadowMesh(objectMesh, objectTransform, lightSource, planeY, planeNormal)
    local shadowMesh = Mesh.new()
    local shadowVertices = {}

    for _, triangle in ipairs(objectMesh.triangles) do
        local worldVertices = {
            objectTransform:transformVector(triangle.v1.position),
            objectTransform:transformVector(triangle.v2.position),
            objectTransform:transformVector(triangle.v3.position)
        }

        local edge1 = worldVertices[2]:subtract(worldVertices[1])
        local edge2 = worldVertices[3]:subtract(worldVertices[1])
        local normal = edge1:cross(edge2):normalize()

        local lightDirection
        if lightSource.type == "directional" then
            lightDirection = lightSource.direction:multiply(-1)
        else
            local center = Vector3.new(
                (worldVertices[1].x + worldVertices[2].x + worldVertices[3].x) / 3,
                (worldVertices[1].y + worldVertices[2].y + worldVertices[3].y) / 3,
                (worldVertices[1].z + worldVertices[2].z + worldVertices[3].z) / 3
            )
            lightDirection = lightSource.position:subtract(center):normalize()
        end

        if normal:dot(lightDirection) < 0.1 then
            local projectedVertices = {}
            local validProjections = 0

            for _, vertex in ipairs(worldVertices) do
                local projection = self:calculateShadowProjection(vertex, lightSource, planeY, planeNormal)
                table.insert(projectedVertices, projection)
                if projection then
                    validProjections = validProjections + 1
                end
            end

            if validProjections == 3 then
                local indices = {}
                for i, projection in ipairs(projectedVertices) do
                    if projection then
                        local vertexIndex = shadowMesh:addVertex(projection.position, planeNormal or Vector3.new(0, 1, 0))
                        table.insert(indices, {index = vertexIndex, intensity = projection.intensity})
                    end
                end

                if #indices == 3 then
                    shadowMesh:addTriangle(indices[1].index, indices[2].index, indices[3].index, nil)

                    shadowMesh.triangles[#shadowMesh.triangles].shadowIntensity = 
                        (indices[1].intensity + indices[2].intensity + indices[3].intensity) / 3
                end
            end
        end
    end

    return shadowMesh
end

function ShadowSystem:renderShadowsForObject(renderer, object, lights, shadowPlanes)
    if not object.castsShadow then return end

    local objectTransform = object:updateTransform()

    for _, light in ipairs(lights) do
        if light.type ~= "ambient" then
            for _, plane in ipairs(shadowPlanes) do
                local shadowMesh = self:generateShadowMesh(
                    object.mesh, 
                    objectTransform, 
                    light, 
                    plane.y, 
                    plane.normal
                )

                if shadowMesh and #shadowMesh.triangles > 0 then

                    local shadowMaterial = Material.new(
                        {r = 0, g = 0, b = 0},
                        {r = 0, g = 0, b = 0},
                        1.0,
                        {r = 0, g = 0, b = 0}
                    )

                    local shadowObject = SceneObject.new(shadowMesh, shadowMaterial)
                    shadowObject.castsShadow = false 
                    shadowObject.receivesShadow = false

                    renderer:renderObjectWithCustomShading(shadowObject, function(triangle, material, worldPos, normal, viewDir)
                        local baseIntensity = triangle.shadowIntensity or self.shadowIntensity
                        local shadowColor = {
                            r = math.floor(20 * baseIntensity),
                            g = math.floor(20 * baseIntensity),
                            b = math.floor(30 * baseIntensity)
                        }
                        return shadowColor
                    end)
                end
            end
        end
    end
end

Renderer = {}
Renderer.__index = Renderer

function Renderer.new()
    local renderer = setmetatable({
        scene = nil,
        zBuffer = {},
        shadowSystem = ShadowSystem.new(),
        triangleRenderer = TriangleRenderer.new(),
        mode = "filled", 
        stats = {trianglesRendered = 0, framesRendered = 0, fps = 0}
    }, Renderer)

    for y = 1, SCREEN_HEIGHT do
        renderer.zBuffer[y] = {}
        for x = 1, SCREEN_WIDTH do
            renderer.zBuffer[y][x] = 100.0 
        end
    end

    return renderer
end

function Renderer:setScene(scene)
    self.scene = scene
end

function Renderer:setMode(mode)
    self.mode = mode
end

function Renderer:drawTriangle(v1, v2, v3, material)
    local lights = self.scene and self.scene.lights or {}
    local camera = self.scene and self.scene.camera or Camera.new()
    self.triangleRenderer:drawTriangle(v1, v2, v3, material, lights, camera)
end

function Renderer:clearScreen(color)
    color = color or {r = 0, g = 0, b = 0}
    screen.setColor(color.r, color.g, color.b)
    screen.fill(1, 1, SCREEN_WIDTH, SCREEN_HEIGHT)
end

function Renderer:clearZBuffer()
    for y = 1, SCREEN_HEIGHT do
        for x = 1, SCREEN_WIDTH do
            self.zBuffer[y][x] = 100.0 
        end
    end
    self.triangleRenderer:clearZBuffer()
end

function Renderer:projectVertex(worldPos, worldNormal, viewMatrix, projMatrix)
    local viewRaw = {
        x = worldPos.x * viewMatrix.m[1] + worldPos.y * viewMatrix.m[2] + worldPos.z * viewMatrix.m[3] + viewMatrix.m[4],
        y = worldPos.x * viewMatrix.m[5] + worldPos.y * viewMatrix.m[6] + worldPos.z * viewMatrix.m[7] + viewMatrix.m[8],
        z = worldPos.x * viewMatrix.m[9] + worldPos.y * viewMatrix.m[10] + worldPos.z * viewMatrix.m[11] + viewMatrix.m[12],
        w = worldPos.x * viewMatrix.m[13] + worldPos.y * viewMatrix.m[14] + worldPos.z * viewMatrix.m[15] + viewMatrix.m[16]
    }

    if viewRaw.z >= -0.1 then return nil end 

    local clip = {
        x = viewRaw.x * projMatrix.m[1] + viewRaw.y * projMatrix.m[2] + viewRaw.z * projMatrix.m[3] + viewRaw.w * projMatrix.m[4],
        y = viewRaw.x * projMatrix.m[5] + viewRaw.y * projMatrix.m[6] + viewRaw.z * projMatrix.m[7] + viewRaw.w * projMatrix.m[8],
        z = viewRaw.x * projMatrix.m[9] + viewRaw.y * projMatrix.m[10] + viewRaw.z * projMatrix.m[11] + viewRaw.w * projMatrix.m[12],
        w = viewRaw.x * projMatrix.m[13] + viewRaw.y * projMatrix.m[14] + viewRaw.z * projMatrix.m[15] + viewRaw.w * projMatrix.m[16]
    }

    if clip.w == 0 then return nil end

    local ndcX, ndcY = clip.x / clip.w, clip.y / clip.w
    local screenX = math.floor(SCREEN_CENTER_X + ndcX * SCREEN_CENTER_X)
    local screenY = math.floor(SCREEN_CENTER_Y - ndcY * SCREEN_CENTER_Y)

    return {
        x = clamp(screenX, 0, SCREEN_WIDTH - 1),
        y = clamp(screenY, 0, SCREEN_HEIGHT - 1),
        z = -viewRaw.z, 
        invW = 1.0 / clip.w,
        posDivW = {x = worldPos.x / clip.w, y = worldPos.y / clip.w, z = worldPos.z / clip.w},
        normalDivW = {x = worldNormal.x, y = worldNormal.y, z = worldNormal.z},
        depthDivW = -viewRaw.z / clip.w,
        original = {position = worldPos, normal = worldNormal}
    }
end

function Renderer:isInShadow(worldPos, lights, shadowCasters)
    for _, light in ipairs(lights) do
        if light.type ~= "ambient" then
            for _, caster in ipairs(shadowCasters) do
                if caster.castsShadow and self:rayIntersectsObject(worldPos, light, caster) then
                    return true, 0.3 
                end
            end
        end
    end
    return false, 1.0 
end

function Renderer:rayIntersectsObject(point, light, object)
    local lightDir
    local lightDistance

    if light.type == "directional" then
        lightDir = light.direction:multiply(-1):normalize()
        lightDistance = 1000 
    elseif light.type == "point" or light.type == "spot" then
        local lightVector = light.position:subtract(point)
        lightDistance = lightVector:length()
        if lightDistance > light.range then return false end
        lightDir = lightVector:normalize()
    else
        return false
    end

    local transform = object:updateTransform()

    local objectCenter = transform:transformVector(Vector3.new(0, 0, 0))
    local toObject = objectCenter:subtract(point)
    local projectedDistance = toObject:dot(lightDir)

    if projectedDistance <= 0 or projectedDistance >= lightDistance then
        return false
    end

    local closestPoint = point:add(lightDir:multiply(projectedDistance))
    local distanceToCenter = closestPoint:subtract(objectCenter):length()

    local boundingRadius = 2.5

    return distanceToCenter <= boundingRadius
end

function Renderer:calculateLighting(position, normal, viewDir, material)
    local finalColor = {
        r = material.emission.r,
        g = material.emission.g,
        b = material.emission.b
    }

    local ambient = {
        r = self.scene.ambientLight.r * material.ambient.r / 255,
        g = self.scene.ambientLight.g * material.ambient.g / 255,
        b = self.scene.ambientLight.b * material.ambient.b / 255
    }

    finalColor.r = finalColor.r + ambient.r
    finalColor.g = finalColor.g + ambient.g
    finalColor.b = finalColor.b + ambient.b

    local shadowCasters = {}
    for _, obj in ipairs(self.scene.objects) do
        if obj.castsShadow then
            table.insert(shadowCasters, obj)
        end
    end

    local inShadow, shadowFactor = self:isInShadow(position, self.scene.lights, shadowCasters)

    for _, light in ipairs(self.scene.lights) do
        local lightDir, lightColor, attenuation = Vector3.new(0, 0, 0), light.color, 1.0

        if light.type == "ambient" then
            lightColor = {
                r = light.color.r * light.intensity,
                g = light.color.g * light.intensity,
                b = light.color.b * light.intensity
            }

            finalColor.r = finalColor.r + material.ambient.r * lightColor.r / 255
            finalColor.g = finalColor.g + material.ambient.g * lightColor.g / 255
            finalColor.b = finalColor.b + material.ambient.b * lightColor.b / 255
        elseif light.type == "directional" then
            lightDir = light.direction:multiply(-1):normalize()
            lightColor = {
                r = light.color.r * light.intensity * shadowFactor,
                g = light.color.g * light.intensity * shadowFactor,
                b = light.color.b * light.intensity * shadowFactor
            }
        elseif light.type == "point" then
            lightDir = light.position:subtract(position):normalize()
            local distance = light.position:subtract(position):length()

            if distance > light.range then goto continue_light end

            attenuation = 1.0 / (light.attenuation.constant + 
                                light.attenuation.linear * distance + 
                                light.attenuation.quadratic * distance * distance)
            lightColor = {
                r = light.color.r * light.intensity * attenuation * shadowFactor,
                g = light.color.g * light.intensity * attenuation * shadowFactor,
                b = light.color.b * light.intensity * attenuation * shadowFactor
            }
        elseif light.type == "spot" then
            lightDir = light.position:subtract(position):normalize()
            local distance = light.position:subtract(position):length()

            if distance > light.range then goto continue_light end

            local theta = lightDir:multiply(-1):dot(light.direction)
            local epsilon = light.cutoff - light.outerCutoff
            local intensity = clamp((theta - light.outerCutoff) / epsilon, 0.0, 1.0)

            attenuation = 1.0 / (light.attenuation.constant + 
                                light.attenuation.linear * distance + 
                                light.attenuation.quadratic * distance * distance)

            lightColor = {
                r = light.color.r * light.intensity * intensity * attenuation * shadowFactor,
                g = light.color.g * light.intensity * intensity * attenuation * shadowFactor,
                b = light.color.b * light.intensity * intensity * attenuation * shadowFactor
            }
        end

        local diff = math.max(normal:dot(lightDir), 0.0)
        local diffuse = {
            r = diff * material.diffuse.r * lightColor.r / 255,
            g = diff * material.diffuse.g * lightColor.g / 255,
            b = diff * material.diffuse.b * lightColor.b / 255
        }

        local specular = {r = 0, g = 0, b = 0}
        if diff > 0 then
            local halfwayDir = lightDir:add(viewDir):normalize()
            local spec = math.pow(math.max(normal:dot(halfwayDir), 0.0), material.shininess)
            specular = {
                r = spec * material.specular.r * lightColor.r / 255,
                g = spec * material.specular.g * lightColor.g / 255,
                b = spec * material.specular.b * lightColor.b / 255
            }
        end

        finalColor.r = finalColor.r + diffuse.r + specular.r
        finalColor.g = finalColor.g + diffuse.g + specular.g
        finalColor.b = finalColor.b + diffuse.b + specular.b

        ::continue_light::
    end

    return {
        r = clamp(math.floor(finalColor.r), 0, 255),
        g = clamp(math.floor(finalColor.g), 0, 255),
        b = clamp(math.floor(finalColor.b), 0, 255)
    }
end

function Renderer:drawTriangleWireframe(p1, p2, p3, color)
    if not (p1 and p2 and p3) then return end

    screen.setColor(color.r, color.g, color.b)
    screen.drawLine(p1.x, p1.y, p2.x, p2.y)
    screen.drawLine(p2.x, p2.y, p3.x, p3.y)
    screen.drawLine(p3.x, p3.y, p1.x, p1.y)
end

function Renderer:drawTriangleFilled(p1, p2, p3, color)
    if not (p1 and p2 and p3) then return end

    local points = {p1, p2, p3}
    table.sort(points, function(a, b) return a.y < b.y end)
    local top, mid, bot = points[1], points[2], points[3]

    screen.setColor(color.r, color.g, color.b)

    local function interpY(y, a, b)
        if b.y == a.y then return {x = a.x, depthDivW = a.depthDivW, invW = a.invW} end
        local t = (y - a.y) / (b.y - a.y)
        return {
            x = lerp(a.x, b.x, t),
            depthDivW = lerp(a.depthDivW, b.depthDivW, t),
            invW = lerp(a.invW, b.invW, t)
        }
    end

    local function drawSpan(y, left, right)
        local x1, x2 = left.x, right.x
        if x2 < x1 then return end

        local startX = math.max(0, math.floor(x1))
        local endX = math.min(SCREEN_WIDTH - 1, math.floor(x2))

        local ld, rd = left.depthDivW, right.depthDivW
        local li, ri = left.invW, right.invW

        for x = startX, endX do
            local denom = (x2 > x1) and ((x - x1) / (x2 - x1)) or 0
            local depthDivW = lerp(ld, rd, denom)
            local invW = lerp(li, ri, denom)

            if invW ~= 0 then
                local depth = depthDivW / invW
                if depth < self.zBuffer[y + 1][x + 1] then
                    self.zBuffer[y + 1][x + 1] = depth
                    screen.drawPixel(x, y)
                end
            end
        end
    end

    for y = math.max(0, top.y), math.min(SCREEN_HEIGHT - 1, mid.y) do
        local left = interpY(y, top, bot)
        local right = interpY(y, top, mid)
        if left.x > right.x then left, right = right, left end
        drawSpan(y, left, right)
    end

    for y = math.max(0, mid.y + 1), math.min(SCREEN_HEIGHT - 1, bot.y) do
        local left = interpY(y, top, bot)
        local right = interpY(y, mid, bot)
        if left.x > right.x then left, right = right, left end
        drawSpan(y, left, right)
    end
end

function Renderer:drawTrianglePixelLighting(p1, p2, p3, triangle, material, viewMatrix, projMatrix, customShadingFunc)
    if not (p1 and p2 and p3) then return end

    local points = {p1, p2, p3}
    table.sort(points, function(a, b) return a.y < b.y end)
    local top, mid, bot = points[1], points[2], points[3]

    local function interpYFull(y, a, b)
        if b.y == a.y then
            return {
                x = a.x,
                invW = a.invW,
                depthDivW = a.depthDivW,
                posDivW = {x = a.posDivW.x, y = a.posDivW.y, z = a.posDivW.z},
                normalDivW = {x = a.normalDivW.x, y = a.normalDivW.y, z = a.normalDivW.z}
            }
        end

        local t = (y - a.y) / (b.y - a.y)
        return {
            x = lerp(a.x, b.x, t),
            invW = lerp(a.invW, b.invW, t),
            depthDivW = lerp(a.depthDivW, b.depthDivW, t),
            posDivW = {
                x = lerp(a.posDivW.x, b.posDivW.x, t),
                y = lerp(a.posDivW.y, b.posDivW.y, t),
                z = lerp(a.posDivW.z, b.posDivW.z, t)
            },
            normalDivW = {
                x = lerp(a.normalDivW.x, b.normalDivW.x, t),
                y = lerp(a.normalDivW.y, b.normalDivW.y, t),
                z = lerp(a.normalDivW.z, b.normalDivW.z, t)
            }
        }
    end

    local function drawScanline(y, left, right)
        local x1, x2 = left.x, right.x
        if x2 < x1 then return end

        local startX = math.max(0, math.floor(x1))
        local endX = math.min(SCREEN_WIDTH - 1, math.floor(x2))

        local li, ld = left.invW, left.depthDivW
        local lp, ln = left.posDivW, left.normalDivW

        local ri, rd = right.invW, right.depthDivW
        local rp, rn = right.posDivW, right.normalDivW

        for x = startX, endX do
            local t = (x2 > x1) and ((x - x1) / (x2 - x1)) or 0

            local invW = lerp(li, ri, t)
            if invW == 0 then goto continue_pixel end

            local depthDivW = lerp(ld, rd, t)
            local depth = depthDivW / invW

            if depth < self.zBuffer[y + 1][x + 1] then
                local posX = lerp(lp.x, rp.x, t) / invW
                local posY = lerp(lp.y, rp.y, t) / invW
                local posZ = lerp(lp.z, rp.z, t) / invW
                local worldPos = Vector3.new(posX, posY, posZ)

                local nx = lerp(ln.x, rn.x, t)
                local ny = lerp(ln.y, rn.y, t)
                local nz = lerp(ln.z, rn.z, t)
                local normal = Vector3.new(nx, ny, nz):normalize()

                local viewDir = self.scene.camera.position:subtract(worldPos):normalize()

                local color
                if customShadingFunc then
                    color = customShadingFunc(triangle, material, worldPos, normal, viewDir)
                else
                    color = self:calculateLighting(worldPos, normal, viewDir, material)
                end

                self.zBuffer[y + 1][x + 1] = depth
                screen.setColor(color.r, color.g, color.b)
                screen.drawPixel(x, y)
            end

            ::continue_pixel::
        end
    end

    for y = math.max(0, top.y), math.min(SCREEN_HEIGHT - 1, mid.y) do
        local left = interpYFull(y, top, bot)
        local right = interpYFull(y, top, mid)
        if left.x > right.x then left, right = right, left end
        drawScanline(y, left, right)
    end

    for y = math.max(0, mid.y + 1), math.min(SCREEN_HEIGHT - 1, bot.y) do
        local left = interpYFull(y, top, bot)
        local right = interpYFull(y, mid, bot)
        if left.x > right.x then left, right = right, left end
        drawScanline(y, left, right)
    end
end

function Renderer:renderObjectWithCustomShading(object, customShadingFunc)
    if not self.scene then return 0 end

    local viewMatrix, projMatrix = self.scene.camera:updateMatrices(SCREEN_WIDTH / SCREEN_HEIGHT)
    local transform = object:updateTransform()
    local material = object.material

    for _, triangle in ipairs(object.mesh.triangles) do
        local wv1 = transform:transformVector(triangle.v1.position)
        local wv2 = transform:transformVector(triangle.v2.position)
        local wv3 = transform:transformVector(triangle.v3.position)

        local wn1 = transform:transformNormal(triangle.v1.normal)
        local wn2 = transform:transformNormal(triangle.v2.normal)
        local wn3 = transform:transformNormal(triangle.v3.normal)

        local edge1 = wv2:subtract(wv1)
        local edge2 = wv3:subtract(wv1)
        local normal = edge1:cross(edge2):normalize()

        local center = Vector3.new(
            (wv1.x + wv2.x + wv3.x) / 3,
            (wv1.y + wv2.y + wv3.y) / 3,
            (wv1.z + wv2.z + wv3.z) / 3
        )

        local viewDir = self.scene.camera.position:subtract(center):normalize()

        if normal:dot(viewDir) > 0 then goto continue_triangle end

        local p1 = self:projectVertex(wv1, wn1, viewMatrix, projMatrix)
        local p2 = self:projectVertex(wv2, wn2, viewMatrix, projMatrix)
        local p3 = self:projectVertex(wv3, wn3, viewMatrix, projMatrix)

        if not (p1 and p2 and p3) then goto continue_triangle end

        self.stats.trianglesRendered = self.stats.trianglesRendered + 1

        if self.mode == "wireframe" then
            self:drawTriangleWireframe(p1, p2, p3, {r = 255, g = 255, b = 255})
        elseif self.mode == "filled" then
            local color
            if customShadingFunc then
                color = customShadingFunc(triangle, material, center, normal, viewDir)
            else
                color = self:calculateLighting(center, normal, viewDir, material)
            end
            self:drawTriangleFilled(p1, p2, p3, color)
        elseif self.mode == "pixel_lighting" then
            local triangleForLighting = {
                v1 = {position = wv1, normal = wn1},
                v2 = {position = wv2, normal = wn2},
                v3 = {position = wv3, normal = wn3}
            }
            self:drawTrianglePixelLighting(p1, p2, p3, triangleForLighting, material, viewMatrix, projMatrix, customShadingFunc)
        end

        ::continue_triangle::
    end

    return self.stats.trianglesRendered
end

function Renderer:renderObject(object, materialOverride)
    local material = materialOverride or object.material
    return self:renderObjectWithCustomShading(object, nil)
end

function Renderer:render()
    if not self.scene then return 0 end

    self:clearScreen({r = 30, g = 30, b = 60})
    self:clearZBuffer()

    self.shadowSystem.shadowPlanes = {}
    for _, obj in ipairs(self.scene.objects) do
        if obj.receivesShadow and obj.position.y < 0 then
            self.shadowSystem:addShadowPlane({y = obj.position.y, normal = Vector3.new(0, 1, 0)})
        end
    end

    local trianglesRendered = 0

    for _, object in ipairs(self.scene.objects) do
        if object.receivesShadow then
            trianglesRendered = trianglesRendered + self:renderObject(object)
        end
    end

    for _, object in ipairs(self.scene.objects) do
        if object.castsShadow then
            self.shadowSystem:renderShadowsForObject(self, object, self.scene.lights, self.shadowSystem.shadowPlanes)
        end
    end

    for _, object in ipairs(self.scene.objects) do
        if not object.receivesShadow then
            trianglesRendered = trianglesRendered + self:renderObject(object)
        end
    end

    self.stats.framesRendered = self.stats.framesRendered + 1

    screen.draw()

    return trianglesRendered
end

local sampleOBJ = [[
v -0.5 -0.5 -0.5
v  0.5 -0.5 -0.5
v  0.5  0.5 -0.5
v -0.5  0.5 -0.5
v -0.5 -0.5  0.5
v  0.5 -0.5  0.5
v  0.5  0.5  0.5
v -0.5  0.5  0.5

f 5 6 7
f 5 7 8

f 1 4 3
f 1 3 2

f 1 5 8
f 1 8 4

f 2 3 7
f 2 7 6

f 4 8 7
f 4 7 3

f 1 2 6
f 1 6 5
]]

local materials = {
    redPlastic = Material.new(
        {r = 200, g = 50, b = 50},
        {r = 255, g = 255, b = 255},
        128.0,
        {r = 50, g = 10, b = 10}
    ),
    emerald = Material.new(
        {r = 80, g = 200, b = 120},
        {r = 255, g = 255, b = 255},
        76.8,
        {r = 5, g = 20, b = 5},
        {r = 0, g = 10, b = 0}
    ),
    floor = Material.new(
        {r = 120, g = 120, b = 130},
        {r = 50, g = 50, b = 50},
        16.0,
        {r = 40, g = 40, b = 45}
    ),
    gold = Material.new(
        {r = 255, g = 215, b = 0},
        {r = 255, g = 255, b = 255},
        51.2,
        {r = 75, g = 60, b = 0}
    ),
    objMaterial = Material.new(
        {r = 150, g = 100, b = 200},
        {r = 255, g = 255, b = 255},
        64.0,
        {r = 30, g = 20, b = 40}
    )
}

local scene = Scene.new()

local cube = SceneObject.new(
    Mesh.cube(1.5, materials.redPlastic),
    materials.redPlastic,
    Vector3.new(-3, 1, 0),
    Vector3.new(0, 0, 0),
    Vector3.new(1, 1, 1),
    true,  
    false  
)

local sphere = SceneObject.new(
    Mesh.sphere(1.2, 20, materials.emerald),
    materials.emerald,
    Vector3.new(3, 0.5, 0),
    Vector3.new(0, 0, 0),
    Vector3.new(1, 1, 1),
    true,  
    false  
)

local objMesh = Mesh.fromOBJ(sampleOBJ, materials.objMaterial, 1.2)
local objObject = SceneObject.new(
    objMesh,
    materials.objMaterial,
    Vector3.new(0, 2, 1),
    Vector3.new(0, 0, 0),
    Vector3.new(1, 1, 1),
    true,
    false
)

local smallSphere = SceneObject.new(
    Mesh.sphere(0.6, 12, materials.gold),
    materials.gold,
    Vector3.new(0, 2.5, -1),
    Vector3.new(0, 0, 0),
    Vector3.new(1, 1, 1),
    true,  
    false  
)

local floor = SceneObject.new(
    Mesh.plane(25, -3, materials.floor),
    materials.floor,
    Vector3.new(0, -3, 0),
    Vector3.new(0, 0, 0),
    Vector3.new(1, 1, 1),
    false, 
    true   
)

scene:addObject(floor)
scene:addObject(cube)
scene:addObject(sphere)
scene:addObject(objObject)
scene:addObject(smallSphere)

local primaryLight = Light.directional(
    Vector3.new(0.3, -1, 0.4),
    {r = 255, g = 250, b = 235},
    1.8
)

local fillLight = Light.point(
    Vector3.new(-3, 4, 3),
    {r = 200, g = 220, b = 255},
    1.2,
    15
)

local rimLight = Light.spot(
    Vector3.new(4, 3, -2),
    Vector3.new(-1, -0.5, 0.5),
    {r = 255, g = 200, b = 150},
    1.0,
    25, 35, 12
)

local ambientLight = Light.ambient(
    {r = 40, g = 45, b = 60},
    7.5
)

scene:addLight(primaryLight)
scene:addLight(fillLight)
scene:addLight(rimLight)
scene:addLight(ambientLight)

scene.camera:setPosition(0, 2, 12)
scene.camera:setTarget(0, 0, 0)

local renderer = Renderer.new()
renderer:setScene(scene)
renderer:setMode("pixel_lighting")

local rotation = Vector3.new(0, 0, 0)
local lightAnimation = 0
local frameCount = 0
local lastFpsUpdate = chip.getTime()
local stats = {frames = 0, lastTime = chip.getTime(), fps = 0}

while true do
    local currentTime = chip.getTime()
    local deltaTime = (currentTime - stats.lastTime) / 1000.0

    if currentTime - lastFpsUpdate >= 1000 then
        stats.fps = frameCount
        frameCount = 0
        lastFpsUpdate = currentTime
    end

    if currentTime - stats.lastTime >= 16 then 
        rotation.x = rotation.x + deltaTime * 0.4
        rotation.y = rotation.y + deltaTime * 0.6
        rotation.z = rotation.z + deltaTime * 0.2

        lightAnimation = lightAnimation + deltaTime

        cube:setRotation(rotation.x, rotation.y, rotation.z)
        sphere:setRotation(rotation.x * 0.5, rotation.y * 1.2, rotation.z * 0.8)
        objObject:setRotation(rotation.x * 0.8, rotation.y * 1.5, rotation.z * 0.3)
        smallSphere:setRotation(rotation.x * 1.5, rotation.y * 0.7, rotation.z * 1.3)

        if fillLight then
            fillLight.position.x = -3 + 2 * math.sin(lightAnimation * 0.3)
            fillLight.position.z = 3 + 1.5 * math.cos(lightAnimation * 0.4)
        end

        if rimLight then
            rimLight.position.y = 3 + 1 * math.sin(lightAnimation * 0.5)

            local angle = lightAnimation * 0.2
            rimLight.position.x = 4 * math.cos(angle)
            rimLight.position.z = -2 + 2 * math.sin(angle)

            local center = Vector3.new(0, 0, 0)
            local newDir = center:subtract(rimLight.position):normalize()
            rimLight.direction = newDir
        end

        smallSphere:setPosition(
            1.5 * math.sin(lightAnimation * 0.8),
            2.5 + 0.5 * math.sin(lightAnimation * 1.2),
            -1 + 0.8 * math.cos(lightAnimation * 0.6)
        )

        renderer:render()
        screen.setColor(255, 0, 0)
        screen.drawPixel(1, 1)
        screen.draw()

        frameCount = frameCount + 1
        stats.lastTime = currentTime

        local t = chip.getTime()
        while true do if (t/1000) + 3 <= (chip.getTime()/1000) then break end end
    end
end
