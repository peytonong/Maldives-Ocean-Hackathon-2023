import path from 'path';
import React, { useEffect } from 'react';

interface iVisualization {
    width: string,
    height: string,
    imgsrc: string,
    staticimg: string,
    statname: string,
    path: string,
}

const VisualizeStat: React.FC<iVisualization> = (props) => {
    let statsHTML;

    useEffect(() => {
        const divElement = document.getElementById('viz1697187810272');
        const vizElement = divElement?.getElementsByTagName('object')[0];

        if (divElement == null || vizElement == null)
            return;

        vizElement.style.width = props.width;
        vizElement.style.height = props.height;

        const scriptElement = document.createElement('script');
        scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
        vizElement.parentNode?.insertBefore(scriptElement, vizElement);
    }, []);

    return (
        <div className='tableauPlaceholder d-flex' id='viz1697187810272' style={{ position: 'relative' }}>
            <noscript>
                <a href='#'>
                    <img alt='Dashboard 1' src={props.imgsrc} style={{ border: 'none' }} />
                </a>
            </noscript>
            <object className='tableauViz' style={{ display: 'none' }}>
                <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
                <param name='embed_code_version' value='3' />
                <param name='site_root' value='' />
                <param name='name' value={props.statname}/>
                <param name='path' value={props.path}/>
                <param name='tabs' value='no' />
                <param name='toolbar' value='hidden' />
                <param name='static_image' value={props.staticimg} />
                <param name='animate_transition' value='yes' />
                <param name='display_static_image' value='yes' />
                <param name='display_spinner' value='no' />
                <param name='display_overlay' value='no' />
                <param name='display_count' value='yes' />
                <param name='language' value='en-US' />
            </object>
        </div>
    );
};

export default VisualizeStat;

