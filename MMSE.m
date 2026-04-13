function mmse = MMSE(xt,x0)
    err = xt - x0;
    mmse = norm(err)/norm(x0);
end