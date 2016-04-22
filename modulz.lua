-----------------------------------------------------
--[[ modulz ]]--
-- Utility function to assist OOP and lazy submodules
-- inside a primary package
-- Usage:
-- (require 'modulz')('primary pkg name')
-----------------------------------------------------

local pl = (require 'pl.import_into')()
pl.stringx.import()

local split = pl.stringx.split

-- create a loading module which
-- with overloaded __index function
-- looks for a file and requires it
local function createLazyloadingModule(name)
   local mod = {
      relmodname = '', -- relative module name for eg:
                       -- if mod name is enigma.feature.Feature
                       -- then relmodname is feature.Feature
      loadedsubmodules = {}  -- contains what submodules have been loaded
   } -- this is the lazy loaded

   function mod:_relativemodname(name)
      if self.relmodname ~= '' then return self.relmodname.."."..name
      else return name end
   end

   if name then mod.relmodname = name end

   local modmt = {}

   -- override __index to lazy load
   -- any missing modules
   function modmt.__index(t, name)
      local mod = require(t:_relativemodname(name))
      if mod then
         rawset(t, name, mod)
         return t[name]
      else
         error(string.format('No module named %s present', name))
      end
   end

   setmetatable(mod, modmt)

   return mod
end

--
local function getOrCreateSubmodule(name, parent)
   assert(type(parent) == 'table', 'Parent is not yet created')
   if not parent.loadedsubmodules[name] then
      parent[name] = createLazyloadingModule(parent:_relativemodname(name))
      parent.loadedsubmodules[name] = true
      return parent[name]
   else
      return parent[name]
   end
end

--
local function constructortbl(metatable)
   local ct = {}
   setmetatable(ct, {
      __index=metatable,
      __newindex=metatable,
      __metatable=metatable,
      __call=function(self, ...)
         return self.new(...)
      end
   })
   return ct
end

--
return function (primarypkgname)

   ------- Helper Function ------

   -- check whether the name in within
   -- primary
   local function withinprimary(name) return name:sub(1, #primarypkgname) == primarypkgname end


   -------- Global OOP Helper Functions --------

   -- primary package in global scope
   _G[primarypkgname] = createLazyloadingModule()

   -- create a submodule (lazy loaded)
   -- automatically create parent if not present
   _G.submodule = function (name)
      assert(withinprimary(name), 'Submodule not inside primary package '..primarypkgname)

      local mod = split(name, '.', 2)[2]
      assert(mod, 'No name for submodules')

      local parent = _G[primarypkgname]

      for _, submod in ipairs(split(mod, '.')) do
         parent = getOrCreateSubmodule(submod, parent)
      end
      return parent
   end

   --
   _G.import = function (_name) local name = _name:replace("\n", ""):replace(" ", "")
      assert(withinprimary(name), 'Submodule not inside primary package '..primarypkgname)
      local mod = split(name, ".", 2)[2]
      assert(mod, 'No name for submodules')

      local parent = _G[primarypkgname]

      local returns = {}
      local mods = split(mod, '.')
      local ismultiimport = false

      for k, submod in ipairs(mods) do
         if submod:sub(1, 1) == '{' then
            if k ~= #mods then
               error(string.format('Multi import not allowed in between, error at [%s] in [%s]', submod, name))
            else
               local multimods = split(submod:sub(2, #submod - 1), ",")
               for _, multimod in ipairs(multimods) do
                  returns[#returns + 1] = parent[multimod]
               end
            end
            ismultiimport = true
            break -- after a multi import, no further import is valid anyway
         else parent = parent[submod]
         end
      end

      if ismultiimport then return table.unpack(returns)
      else return parent
      end
   end

   --
   _G.klazz = function (name, parentname)
      assert(type(name) == 'string', "Class's name should be string")
      assert(withinprimary(name), 'Both class and parent class should be with primary pkg = '..primarypkgname)

      local mt = { __typename = name }

      mt.__index = mt

      mt.__factory =
         function ()
            local self = {}
            setmetatable(self, mt)
            return self
         end

      mt.__init =
         function () end

      mt.new =
         function(...)
            local self = mt.__factory()
            self:__init(...)
            return self
         end

      if not parentname then
         return constructortbl(mt)
      elseif withinprimary(parentname) then
         local parent = import(parentname)
         local inhmt = { __index = parent } -- NOTE: This is where modifications for multiple inheritance
                                            -- should be done
         inhmt.__metatable = inhmt
         setmetatable(mt, inhmt)
         return constructortbl(mt), parent
      else
         error('Parent name present but not within primary package')
      end
   end

   return _G[primarypkgname]

end