function [violation, W] = fhals_update(W, HHt, AHt)

[m, n] = size(W);
violation = 0;

for t = 1 : n
    for i = 1 : m
        gradient = -AHt(i, t);

        for r = 1 : n
            gradient += HHt(t, r) * W(i, r);
        end

        if W(i, t) == 0
            projected_gradient = min(0.0, gradient);
        else
            projected_gradient = gradient;
        end

        violation += abs(projected_gradient);

        if HHt(t, t) != 0
            W(i, t) = max(0.0, W(i, t) - gradient / HHt(t, t));
        end
    end
end

end % function