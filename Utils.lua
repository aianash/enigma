local Utils = {}

--
function Utils.kldirichlet(phimP, phimQ)
   phimP0 = torch.sum(phimP)
   phimQ0 = torch.sum(phimQ)

   return (cephes.lgam(phimP0) - cephes.lgam(phimQ0)
            - torch.sum(cephes.lgam(phimP) - cephes.lgam(phimQ))
            + (phimP - phimQ):double():dot(cephes.digamma(phimP) - cephes.digamma(phimQ)))
end

--
function Utils.klgamma(pa, pb, qa, qb)
   return torch.sum(
              (pa - qa):double():cmul(cephes.digamma(pa))
            - cephes.lgam(pa) + cephes.lgam(qa)
            + (torch.log(pb) - torch.log(qb)):cmul(qa):double()
            + (pb - qb):cmul(pa):cdiv(pb):double())
end

-- Calculate the nearest positive definite matrix
function Utils.posdefify(M, ev)
   local e, V = torch.symeig(M, 'V')
   local n = e:size(1)
   local ev = ev or 1e-7
   local eps = ev * math.abs(e[n])

   if e[1] < eps then
      e[e:lt(eps)] = eps
      local old = torch.diag(M)
      local Mnew = V * torch.diag(e) * V:t()
      local D = old:cmax(eps):cdiv(torch.diag(Mnew)):sqrt()
      M:copy(torch.diag(D) * Mnew * torch.diag(D))
   end
end

--
function Utils.logdet(M)
   local status, logdet = pcall(function()
      return 2 * torch.sum(torch.log(torch.diag(torch.potrf(M, 'U'))))
   end)

   if not status then
      Utils.posdefify(M)
      logdet = 2 * torch.sum(torch.log(torch.diag(torch.potrf(M, 'U'))))
   end
   return logdet
end

--
function Utils.inverse(M, definite, ev)
   local definite = definite or true
   local status, inv = pcall(function ()
      return torch.inverse(M)
   end)
   local ev = ev or 1e-7

   if not status then
      local u, s, v = torch.svd(M)
      local tol = ev * M:size(1) * s[1]
      s[s:lt(tol)] = 0
      s:pow(-1)
      s[s:eq(math.huge)] = 0
      inv = u * torch.diag(s) * v:t()
      print(string.format("INV = %s", inv))
   end

   if definite then Utils.posdefify(inv) end
   return inv
end

-- from fblualib/fb/util/data.lua , copied here because fblualib is not rockspec ready yet.
-- deepcopy routine that assumes the presence of a 'clone' method in user
-- data should be used to deeply copy. This matches the behavior of Torch
-- tensors.
function Utils.deepcopy(x)
    local typename = type(x)
    if typename == "userdata" then
        return x:clone()
    end
    if typename == "table" then
        local retval = { }
        for k,v in pairs(x) do
            retval[Utils.deepcopy(k)] = Utils.deepcopy(v)
        end
        return retval
    end
    return x
end

return Utils